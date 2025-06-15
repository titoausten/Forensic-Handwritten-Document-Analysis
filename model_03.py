# -*- coding: utf-8 -*-
"""
Created on Mon May 19 2025

@author: JKimeli
"""


import os
import cv2
from collections import defaultdict
from glob import glob
import shutil
import numpy as np
from PIL import Image
import fitz  
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from datetime import datetime
import sys
from timm import create_model

#pip install timm

input_dir = "Training"
output_dir = "Preprocessed"
image_output_root = "ImageData"
epochsAllowed = 50
#all available images per writer
use_all_available_images_per_writer = True
images_per_writer = 6
# ------------------------------
# 1. Utility Functions
# ------------------------------

def get_all_pdf_paths(base_folder='Training'):
    """
    Collect all PDF file paths recursively in the 'Training' directory.
    Organizes them by subfolder (e.g., paper/tablet) and writer.
    """
    pdf_paths = {}
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.pdf'):
                writer_id = os.path.basename(root)
                full_path = os.path.join(root, file)
                if writer_id not in pdf_paths:
                    pdf_paths[writer_id] = []
                pdf_paths[writer_id].append(full_path)
    return pdf_paths

def convert_all_pdfs_to_images(input_dir="Training", image_output_root="ImageData", dpi=300):
    """
    Converts all PDFs in 'input_dir' (e.g., 'Training/paper/', 'Training/tablet/')
    into images and saves them to 'output_dir' with structure:
    - output_dir/paper_images/filename_pageX.jpg
    - output_dir/tablet_images/filename_pageX.jpg
    """
    subfolders = ["paper", "tablet"]

    for sub in subfolders:
        input_subdir = os.path.join(input_dir, sub)
        output_subdir = os.path.join(image_output_root, f"{sub}_images")

        os.makedirs(output_subdir, exist_ok=True)

        for file in os.listdir(input_subdir):
            if not file.lower().endswith(".pdf"):
                continue

            pdf_path = os.path.join(input_subdir, file)
            doc = fitz.open(pdf_path)

            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = img[:, :, :3]  # drop alpha if present

                # Construct output filename
                base_name = os.path.splitext(file)[0]
                img_filename = f"{base_name}_page{i+1}.jpg"
                img_path = os.path.join(output_subdir, img_filename)

                cv2.imwrite(img_path, img)


# ------------------------------
# 2. Dataset + Pair Generator
# ------------------------------

class FHDAImagePairDataset(Dataset):
    """
    Dataset class for loading image pairs for verification.
    """
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1, img2 = self.image_pairs[idx]
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def generate_pairs(image_dict, max_per_writer=10):
    """
    Generating image pairs (same and different writers).
    """
    writers = list(image_dict.keys())
    pairs = []
    labels = []

    for writer in writers:
        imgs = image_dict[writer]
        if not use_all_available_images_per_writer:
            imgs = imgs[:images_per_writer]
            
        if len(imgs) < 2:
            # Skipping if the writer has less than 2 images
            continue  
        # Same writer pairs
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j]))
                labels.append(1)

        # Different writer pairs
        for _ in range(len(imgs)):
            other_writer = np.random.choice([w for w in writers if w != writer])
            img1 = random.choice(imgs)
            img2 = random.choice(image_dict[other_writer])
            pairs.append((img1, img2))
            labels.append(0)
    print(f"Generated {len(pairs)} pairs from {len(writers)} writers")
    save_pairs_to_csv(pairs, labels)
    return pairs, labels

# ------------------------------
# 3. Siamese Network
# ------------------------------
"""
class SiameseNetwork(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super(SiameseNetwork, self).__init__()

        # Load model from timm without classifier head (feature extractor)
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.embedding_dim = self.backbone.num_features

        # Comparison (Fully connected) layer
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward_one(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.fc(combined)
"""
class SiameseNetworkComparison(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super(SiameseNetwork, self).__init__()

        # Use TIMM to create the model (automatically handles all modern backbones)
        base_model = create_model(backbone_name, pretrained=True, num_classes=0)  # num_classes=0 removes classifier
        self.feature_extractor = base_model

        feature_dim = base_model.num_features  # Get output feature dim of backbone

        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        combined = torch.cat((f1, f2), dim=1)
        out = self.fc(combined)
        return out
    
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        base_model = resnet18(pretrained=True)
        # removing final classification
        base_model.fc = nn.Identity()  
        self.feature_extractor = base_model
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        combined = torch.cat((f1, f2), dim=1)
        out = self.fc(combined)
        return out

# ------------------------------
# 4. Main Execution Pipeline
# ------------------------------
def preprocess_all_images(image_root, output_dir="Preprocessed"):
    """
    Preprocess all images under `image_root` (e.g., Paper_images/ and Tablet_images/)
    and save them into structured Preprocessed/{writer_id}/ folders.
    """
    if os.path.exists(output_dir):
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    image_dict = {}

    for writer_folder in os.listdir(image_root):  
        full_folder_path = os.path.join(image_root, writer_folder)
        if not os.path.isdir(full_folder_path):
            continue

        for img_file in os.listdir(full_folder_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue

            # using folder name as writer ID
            writer_id = writer_folder
            if "tablet_images" in writer_folder.lower() or "paper_images" in writer_folder.lower():
                writer_id = img_file.split('_')[0] + '_' + img_file.split('_')[1]
            else:
                writer_id = writer_folder
                
            if writer_id not in image_dict:
                image_dict[writer_id] = []

            img_path = os.path.join(full_folder_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                processed = further_style_preprocessing(img)
            except Exception as e:
                print(f"Error preprocessing {img_path}: {e}")
                continue

            # Save processed image
            writer_output_dir = os.path.join(output_dir, writer_id)
            os.makedirs(writer_output_dir, exist_ok=True)
            output_path = os.path.join(writer_output_dir, img_file)
            cv2.imwrite(output_path, processed)

            image_dict[writer_id].append(processed)

    return image_dict


def further_style_preprocessing(image):
    """
    Preprocess image: grayscale, threshold,
    morphological filtering, and contour-based cropping.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to emphasize handwriting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 15)

        # Morphological operations to remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get bounding box around all contours (handwritten region)
            x_min = min([cv2.boundingRect(c)[0] for c in contours])
            y_min = min([cv2.boundingRect(c)[1] for c in contours])
            x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
            y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

            cropped = image[y_min:y_max, x_min:x_max]
            return cropped
        else:
            return image  # fallback: return original image if no contours
    except Exception as e:
        print(f"Error in notebook-style preprocessing: {e}")
        return image



#================IAM HANDWRITING PRE-PROCESSING==================================
"""
    1. Gathering all image files with extensions .png, .jpg, or .jpeg
    
    2. Extracting writerIds from the image filenames - indicates authorship.
    
    3. Then reorganized them into subfolders for each images of same writer.
    
    (Each subfolder corresponds to a unique writerId).
    
    4. Images are resized to be 224×224.
    
    5. Applied augmentations through the PyTorch transforms.
    
    6. Applying pair generation to be used for training and evaluation
    
"""



def reorganize_iam_dataset(source_IAM_dir='IAM_raw', output_IAM_dir='IAM_Images'):
    """
    Reorganizing the IAM images into writer-specific folders.
    
    Args:
        source_IAM_dir (str): Directory containing all IAM images in flat structure.
        output_IAM_dir (str): Directory where images will be organized per writer.
    """
    if not os.path.exists(output_IAM_dir):
        os.makedirs(output_IAM_dir)

    files = [f for f in os.listdir(source_IAM_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for file in files:
        # Extract writer ID from filename (e.g., a01-000u-00.png => a01-000u)
        parts = file.split('-')
        if len(parts) > 1:
            if len(parts) < 3:
                continue  
            # skipping the malformed filenames
            writer_id = f"{parts[0]}-{parts[1]}"
    
            writer_dir = os.path.join(output_IAM_dir, writer_id)
            os.makedirs(writer_dir, exist_ok=True)
    
            src_path = os.path.join(source_IAM_dir, file)
            dst_path = os.path.join(writer_dir, file)
    
            shutil.copy2(src_path, dst_path)
            
        else:
            print("No image pairs found. Please check the dataset.")
            

    print(f"Reorganized {len(files)} images into writer-specific folders at '{output_dir}'")


def reorganize_cvl_dataset(source_CVL_dir='CVL_raw', output_CVL_dir='CVL_Images'):
    """
    Reorganizing the CVL images into writer-specific folders.
    
    Args:
        source_cvl_dir (str): Directory containing all cvl images in flat structure.
        output_CVL_dir (str): Directory where images will be organized per writer.
    """
    if not os.path.exists(output_CVL_dir):
        os.makedirs(output_CVL_dir)

    files = [f for f in os.listdir(source_CVL_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

    for file in files:
        # Extract writer ID from filename (e.g., 0196-1-cropped.tiff => 0196)
        parts = file.split('-')
        if len(parts) > 1:

            writer_id = f"{parts[0]}"
    
            writer_dir = os.path.join(output_CVL_dir, writer_id)
            os.makedirs(writer_dir, exist_ok=True)
    
            src_path = os.path.join(source_CVL_dir, file)
            dst_path = os.path.join(writer_dir, file)
    
            shutil.copy2(src_path, dst_path)
            
        else:
            print("No image pairs found. Please check the dataset.")
            

    print(f"Reorganized {len(files)} images into writer-specific folders at '{output_dir}'")
    

def load_iam_data(iam_root, max_per_writer=10):
    """
    Load IAM dataset images organized by writer folders.
    
    Parameters:
        iam_root (str): Root directory path containing IAM writer subfolders.
        max_per_writer (int): Maximum number of images to load per writer.
    
    Returns:
        dict: A dictionary with writer IDs as keys and lists of grayscale images as values.
    
    Load IAM dataset images organized by writer folders.
    Supports .png, .jpg, and .jpeg formats.
    """
    image_dict = defaultdict(list)

    if not os.path.exists(iam_root):
        print(f"IAM path does not exist: {iam_root}")
        return image_dict

    for writer_id in os.listdir(iam_root):
        writer_folder = os.path.join(iam_root, writer_id)
        if not os.path.isdir(writer_folder):
            continue

        image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            image_paths.extend(glob(os.path.join(writer_folder, ext)))

        image_paths = sorted(image_paths)[:max_per_writer]

        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                image_dict[writer_id].append(img)

    print(f"IAM dataset loaded: {len(image_dict)} writers, total {sum(len(v) for v in image_dict.values())} images.")
    return image_dict


def save_results_to_csv(accuracy, auc, dataset_name="IAM + Handwriting", model_name="Siamese_ResNet", epochs=epochsAllowed,
                        notes="", output_file="results_log.csv"):
    """
    Saving evaluation metrics and experiment details to a CSV file.
    """
    header = ['Timestamp', 'Dataset', 'Model', 'Epochs', 'Accuracy', 'AUC', 'Notes']
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [now, dataset_name, model_name, epochs, round(accuracy, 4), round(auc, 4), notes]

    try:
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results: {e}")


def save_pairs_to_csv(pairs, labels, filename="pairs_log.csv"):
    """
    Saving generated image pairs and labels to a CSV file in the same directory as the script.

    Args:
        pairs (list): List of image path pairs.
        labels (list): List of labels (1 for same writer, 0 for different writers).
        filename (str): Name of the CSV file (default is pairs_log.csv).
    """
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, filename)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image1', 'Image2', 'Label', 'LabelMeaning'])

        for (img1, img2), label in zip(pairs, labels):
            # If image is a path, keep it as is; otherwise try to extract .filename
            img1_path = img1 if isinstance(img1, str) else getattr(img1, 'filename', 'unknown')
            img2_path = img2 if isinstance(img2, str) else getattr(img2, 'filename', 'unknown')
            label_meaning = "Same Writer" if label == 1 else "Different Writers"
            writer.writerow([img1_path, img2_path, label, label_meaning])


    print(f"Saved {len(pairs)} pairs to {csv_path}")


def visualize_and_show_pair(pairs, labels, num_samples=10):
    sampled = random.sample(list(zip(pairs, labels)), num_samples)
    for idx, ((img1, img2), label) in enumerate(sampled):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img1, cmap='gray')
        axes[0].set_title("Image 1")
        axes[0].axis('off')

        axes[1].imshow(img2, cmap='gray')
        axes[1].set_title("Image 2")
        axes[1].axis('off')

        plt.suptitle(f"Pair {idx+1} - Label: {'Same Writer' if label == 1 else 'Different Writers'}")
        plt.tight_layout()
        plt.show()
  


# ------------------------------
# 5. Training + Evaluation Utils
# ------------------------------

def train_siamese(model, dataloader, criterion, optimizer, device, num_epochs=epochsAllowed):
    model.to(device)
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for img1, img2, labels in dataloader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    return train_losses, train_accuracies


def evaluate_model(model, dataloader, device, export_csv_path=None):
    model.eval()
    preds_list = []
    labels_list = []
    probs_list = []

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            outputs = model(img1, img2).squeeze(1)
            probs = torch.sigmoid(outputs)

            preds_list.extend((probs > 0.5).cpu().numpy())
            probs_list.extend(probs.cpu().numpy())
            labels_list.extend(labels.numpy())

    acc = accuracy_score(labels_list, preds_list)
    roc_auc = roc_auc_score(labels_list, probs_list)
    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    if export_csv_path:
        df = pd.DataFrame({
            'True_Label': labels_list,
            'Predicted_Probability': probs_list,
            'Predicted_Label': preds_list
        })
        df.to_csv(export_csv_path, index=False)
        print(f"Prediction results saved to {export_csv_path}")

    plot_roc_curve(labels_list, probs_list)
    return acc, roc_auc


def evaluate_current_model(model, dataloader, device, export_csv_path=None):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            outputs = model(img1, img2).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_outputs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    preds = (all_outputs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    auc_score = roc_auc_score(all_labels, all_outputs)
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)

    if export_csv_path:
        pd.DataFrame({
            "Label": all_labels,
            "Prob": all_outputs,
            "Pred": preds
        }).to_csv(export_csv_path, index=False)

    return acc, auc_score, (fpr, tpr)


def plot_roc_curve(true_labels, pred_probs):
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def plot_training_curves(losses, accuracies):
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_model_comparisons_old(results):
    plt.figure(figsize=(12, 5))

    # Accuracy bar chart
    plt.subplot(1, 2, 1)
    plt.bar(results["Backbone"], results["Train Accuracy"], color='skyblue')
    plt.title("Test Accuracy by Backbone")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)

    # ROC AUC bar chart
    plt.subplot(1, 2, 2)
    plt.bar(results["Backbone"], results["Train AUC"], color='lightgreen')
    plt.title("ROC AUC by Backbone")
    plt.ylabel("AUC Score")
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()


def plot_model_comparisons(results):
    import matplotlib.pyplot as plt

    backbones = results["Backbone"]

    # Accuracy and AUC Bar Charts 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(backbones, results["Train Accuracy"], color='skyblue')
    plt.title("Test Accuracy by Backbone")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=30)

    plt.subplot(1, 2, 2)
    plt.bar(backbones, results["Train AUC"], color='lightgreen')
    plt.title("ROC AUC by Backbone")
    plt.ylabel("AUC Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    # Combined Training Loss Curve 
    plt.figure(figsize=(8, 6))
    for backbone, loss_curve in zip(backbones, results["Loss Curve"]):
        plt.plot(loss_curve, label=backbone)
    plt.title("Training Loss Curve Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Combined ROC Curve 
    plt.figure(figsize=(8, 6))
    for backbone, roc in zip(backbones, results["ROC"]):
        fpr, tpr = roc
        auc_score = roc_auc_score  
        plt.plot(fpr, tpr, label=f"{backbone}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def train_and_evaluate_models_old(backbone_list, image_pairs, labels, transform, device):
    results = {
        "Backbone": [],
        "Train Accuracy": [],
        "Train AUC": [],
        "Loss Curve": [],
        "Acc Curve": []
    }

    # Train-test split
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        image_pairs, labels, test_size=0.2, random_state=42)

    # Optionally limiting data size for speed 
    #train_pairs, train_labels = train_pairs[:1000], train_labels[:1000]
    #test_pairs, test_labels = test_pairs[:300], test_labels[:300]

    # Dataset and loader setup
    train_dataset = FHDAImagePairDataset(train_pairs, train_labels, transform)
    test_dataset = FHDAImagePairDataset(test_pairs, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)

    for backbone in backbone_list:
        print(f"\nTraining with backbone: {backbone}")
        model = SiameseNetwork(backbone_name=backbone).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_losses, train_accuracies = train_siamese(
            model, train_loader, criterion, optimizer, device, num_epochs=5)

        # Evaluate
        acc, auc = evaluate_model(
            model, test_loader, device,
            export_csv_path=f"predictions_{backbone}.csv")

        # Save results
        results["Backbone"].append(backbone)
        results["Train Accuracy"].append(acc)
        results["Train AUC"].append(auc)
        results["Loss Curve"].append(train_losses)
        results["Acc Curve"].append(train_accuracies)

    return results
    

    
 
    
def main():
    # Step 1: Load PDFs and convert to images
    
    print("Loading and preprocessing images...")
    if not os.path.exists(input_dir):
        raise FileNotFoundError("The 'Training' folder was not found. Please make sure it exists in the working directory.")
    pdf_dict = get_all_pdf_paths(input_dir)
    if not pdf_dict:
        raise ValueError("No PDF files found. Check if the 'Training' folder has the correct structure.")


    print("Converting PDFs to images...")
    convert_all_pdfs_to_images(input_dir, image_output_root)
    
    image_dict = preprocess_all_images(image_output_root)
    print(f"Images preprocessed {[len(v) for v in image_dict.values()]}")
    for writer_id, images in image_dict.items():
        print(f"Writer ID: {writer_id} - Number of Images: {len(images)}")
     
    # Stop further execution
    sys.exit("Stopped execution after examining image_dict.")

    """
    if not os.path.exists('IAM_raw'):
        raise FileNotFoundError("The 'IAM_raw' folder was not found. Please make sure it exists in the working directory.")
    
    print("Loading and preprocessing IAM Handwriting images...")
    reorganize_iam_dataset('IAM_raw', 'IAM_Images')
    
    
    image_dict_initial = load_iam_data('IAM_Images')
    
    image_dict = preprocess_all_images('IAM_Images', 'Preprocessed')
    print(f"Images preprocessed {[len(v) for v in image_dict.values()]}")
    

    # Optionally prefix IAM writer IDs to avoid ID collision
    iam_dict_prefixed = {f'iam_{k}': v for k, v in image_dict.items()}
    
    # Merge both datasets
    image_dict.update(iam_dict_prefixed)
    """
    
    """
    if not os.path.exists('CVL_raw'):
        raise FileNotFoundError("The 'CVL_raw' folder was not found. Please make sure it exists in the working directory.")
    
    print("Loading and preprocessing CVL Handwriting images...")
    reorganize_cvl_dataset('CVL_raw', 'CVL_Images')
    
    image_dict = preprocess_all_images('CVL_Images', 'Preprocessed')
    print(f"Images preprocessed {[len(v) for v in image_dict.values()]}")
    

    # Optionally prefix IAM writer IDs to avoid ID collision
    iam_dict_prefixed = {f'cvl_{k}': v for k, v in image_dict.items()}
    
    # Merge both datasets
    image_dict.update(iam_dict_prefixed)
    """
    
    # Step 2: Generate image pairs
    print("Generating image pairs...")
    pairs, labels = generate_pairs(image_dict, images_per_writer)
    print(f"Generated {len(pairs)} pairs")
    
    #Showing about 10 random pairs to inspect preprocessing results
    #visualize_and_show_pair(pairs, labels, num_samples=10)
    
    # Step 3: Train/test split
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42)
    
    #In production version we shall comment line below as it only limits to 1000 data to make it train faster
    #train_pairs, train_labels = train_pairs[:1000], train_labels[:1000]


    # Step 4: Preparing the datasets
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        
        transforms.ToTensor()
    ])
    
    
    # Step 5: Creating dataset and dataloader
    train_dataset = FHDAImagePairDataset(train_pairs, train_labels, transform)
    test_dataset = FHDAImagePairDataset(test_pairs, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)


    # Step 6: Initializing and training the model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    
    # Step 7: Train the model
    #train_siamese(model, train_loader, criterion, optimizer, device, num_epochs=epochsAllowed)
    
    # Train
    train_losses, train_accuracies = train_siamese(model, train_loader, criterion, optimizer, device, num_epochs=10)
    
    # Plot training performance
    plot_training_curves(train_losses, train_accuracies)
    
    # Evaluate
    evaluate_model(model, train_loader, device, export_csv_path="predictions.csv")



    """
    backbone_list = [
    "resnet18",
    "resnet26d",
    "res2net50_26w_4s",
    "resnext50_32x4d",
    "tf_efficientnet_b0"
    ]


    results = {
        "Backbone": [],
        "Train Accuracy": [],
        "Train AUC": [],
        "Loss Curve": [],
        "Acc Curve": [],
        "ROC": []
    }
    
    for backbone in backbone_list:
       print(f"\nTraining with backbone: {backbone}")
       model = SiameseNetwork(backbone_name=backbone).to(device)

       criterion = nn.BCEWithLogitsLoss()
       optimizer = optim.Adam(model.parameters(), lr=1e-4)

       train_losses, train_accuracies = train_siamese(
           model, train_loader, criterion, optimizer, device, num_epochs=epochsAllowed)

       acc, auc, roc_data = evaluate_current_model(
           model, test_loader, device,
           export_csv_path=f"predictions_{backbone}.csv")

       results["Backbone"].append(backbone)
       results["Train Accuracy"].append(acc)
       results["Train AUC"].append(auc)
       results["Loss Curve"].append(train_losses)
       results["Acc Curve"].append(train_accuracies)
       results["ROC"].append(roc_data)
       
    #results = train_and_evaluate_models(backbones_to_test, pairs, labels, transform, device)
    plot_model_comparisons(results)
    
    """

if __name__ == "__main__":
    main()

