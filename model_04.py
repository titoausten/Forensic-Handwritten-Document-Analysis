# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 18:11:46 2025

@author: JKimeli
"""


import os
import cv2
from collections import defaultdict
from glob import glob
import shutil
import numpy as np
#from PIL import Image
import fitz  
import random
import csv
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from datetime import datetime
#import sys
#from timm import create_model

#pip install timm

input_dir = "Training"
output_dir = "Preprocessed"
image_output_root = "ImageData"
epochsAllowed = 20
#all available images per writer
use_all_available_images_per_writer = True
images_per_writer = 6
pairs_dict = defaultdict(list)
best_val_acc = 0.0
best_model_path = "best_model.pth"
# ------------------------------
# 1. Utility Functions
# ------------------------------

def get_all_pdf_paths(base_folder='Training'):
    """
    Collect all PDF file paths recursively in the 'Training' directory.
    Organizes them by subfolder (e.g., paper/tablet) and writer.
    """
    pdf_paths = defaultdict(list)
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


# PDF to Image Conversion with High DPI
def convert_pdf_to_images(input_dir, output_dir, dpi=300):
    os.makedirs(output_dir, exist_ok=True)
    
    for pdf_file in glob(os.path.join(input_dir, "*.pdf")):
        doc = fitz.open(pdf_file)
        base_name = os.path.splitext(os.path.basename(pdf_file))[0]

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = img[:, :, :3] if pix.n == 4 else img  # Remove alpha channel if present

            output_path = os.path.join(output_dir, f"{base_name}_page{i+1}.jpg")
            cv2.imwrite(output_path, img)

# Cropping to Handwriting Area using Contours
def crop_handwriting_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x_min, y_min, x_max, y_max = min([cv2.boundingRect(c)[0] for c in contours]), \
                                     min([cv2.boundingRect(c)[1] for c in contours]), \
                                     max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours]), \
                                     max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])
        return image[y_min:y_max, x_min:x_max]
    
    return image  # Fallback if no contours detected

# Rotation Correction using Hough Line Transform
def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        angle = np.median([line[0][1] for line in lines]) * 180 / np.pi
        rotation_matrix = cv2.getRotationMatrix2D((gray.shape[1] // 2, gray.shape[0] // 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (gray.shape[1], gray.shape[0]))

    return image  # Fallback if no rotation detected

# Gridline Removal using Fourier Transform
def remove_gridlines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    rows, cols = gray.shape
    center_row, center_col = rows // 2, cols // 2
    fshift[center_row-30:center_row+30, center_col-30:center_col+30] = 0
    filtered = np.fft.ifft2(np.fft.ifftshift(fshift))
    return np.uint8(np.abs(filtered))

# Ink Standardization (Convert Blue Ink to Black)
def standardize_ink_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array([90, 50, 50]), np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    image[mask > 0] = [0, 0, 0]  # Convert blue ink to black
    return image

# Adaptive Inversion for Black-Background Documents
def adaptive_inversion(image):
    avg_brightness = np.mean(image)
    return cv2.bitwise_not(image) if avg_brightness < 50 else image  # Invert if too dark


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

def generate_pairs(image_dict_loaded, max_per_writer=10):
    """
    Generate image pairs (same and different writers) and log filenames.
    Assumes each entry in image_dict[writer] is a tuple: (image_tensor, image_path)
    """
    im_dict = image_dict_loaded
    if pairs_dict:
        im_dict = pairs_dict
    
    writers = list(im_dict.keys())
    pairs = []
    labels = []
    pair_logs = []

    for writer in writers:
        imgs = im_dict[writer]
        if not use_all_available_images_per_writer:
            imgs = imgs[:images_per_writer]

        if len(imgs) < 2:
            continue  # Skipping writers with fewer than 2 images

        # Same writer pairs
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                img1, path1 = imgs[i]
                img2, path2 = imgs[j]
                pairs.append((img1, img2))
                labels.append(1)
                pair_logs.append((os.path.basename(path1), os.path.basename(path2), 1, "Same Writer"))

        # Different writer pairs
        for _ in range(len(imgs)):
            other_writer = random.choice([w for w in writers if w != writer])
            img1, path1 = random.choice(imgs)
            img2, path2 = random.choice(im_dict[other_writer])
            pairs.append((img1, img2))
            labels.append(0)
            pair_logs.append((os.path.basename(path1), os.path.basename(path2), 0, "Different Writer"))

    print(f"Generated {len(pairs)} pairs from {len(writers)} writers")
    save_pairs_to_csv(pair_logs)  # ← pass logs, not just pairs and labels
    return pairs, labels

# ------------------------------
# 3. Siamese Network
# ------------------------------
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

    image_dict = defaultdict(list)
    pairs_dict = defaultdict(list)

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

            img_path = os.path.join(full_folder_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            try:
                image = crop_handwriting_region(img)
                image = correct_skew(image)
                image = remove_gridlines(image)
                image = standardize_ink_color(image)
                processed = adaptive_inversion(image)
                #processed = further_style_preprocessing(image)
            except Exception as e:
                print(f"Error preprocessing {img_path}: {e}")
                continue

            # Save processed image
            writer_output_dir = os.path.join(output_dir, writer_id)
            os.makedirs(writer_output_dir, exist_ok=True)
            output_path = os.path.join(writer_output_dir, img_file)
            cv2.imwrite(output_path, processed)

            image_dict[writer_id].append(processed)
            pairs_dict[writer_id].append((processed, output_path))

    return image_dict, pairs_dict


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
        # Extracting writer ID from filename (e.g., 0196-1-cropped.tiff => 0196)
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


def save_pairs_to_csv(pair_logs, filename="pairs_log.csv"):
    """
    Saving image pair filenames and labels to CSV for traceability.
    """
    df = pd.DataFrame(pair_logs, columns=["Image1", "Image2", "Label", "LabelMeaning"])
    df.to_csv(filename, index=False)
    print(f"Pair logs saved to {filename}")


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
    best_acc = 0.0

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

        # Save best model based on training accuracy
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), best_model_path)
            #print(f"Best model updated at Epoch {epoch+1} with accuracy: {accuracy:.4f}")
        best_val_acc = best_acc
    print(f"Training complete. Best accuracy: {best_acc:.4f}")

    return train_losses, train_accuracies


def evaluate_model(model, dataloader, device, export_csv_path=None):
    #model.eval()
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
    #model.eval()
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
        #auc_score = roc_auc_score  
        plt.plot(fpr, tpr, label=f"{backbone}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
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
    #convert_pdf_to_images(input_dir, image_output_root, dpi=300)

    image_dict, pairs_dict = preprocess_all_images(image_output_root)
    print(f"Images preprocessed {[len(v) for v in image_dict.values()]}")
    
    #for writer_id, images in image_dict.items():
        #print(f"Writer ID: {writer_id} - Number of Images: {len(images)}")
     

    """
    # For IAM HAndwriting Dataset
    if not os.path.exists('IAM_raw'):
        raise FileNotFoundError("The 'IAM_raw' folder was not found. Please make sure it exists in the working directory.")
    
    print("Loading and preprocessing IAM Handwriting images...")
    reorganize_iam_dataset('IAM_raw', 'IAM_Images')
    
    
    image_dict_initial = load_iam_data('IAM_Images')
    
    image_dict, pairs_dict = preprocess_all_images('IAM_Images', 'Preprocessed')
    print(f"Images preprocessed {[len(v) for v in image_dict.values()]}")
    

    # Optionally prefix IAM writer IDs to avoid ID collision
    iam_dict_prefixed = {f'iam_{k}': v for k, v in image_dict.items()}
    
    # Merge both datasets
    image_dict.update(iam_dict_prefixed)
    """
    
    """
    # For CVL Dataset
    if not os.path.exists('CVL_raw'):
        raise FileNotFoundError("The 'CVL_raw' folder was not found. Please make sure it exists in the working directory.")
    
    print("Loading and preprocessing CVL Handwriting images...")
    reorganize_cvl_dataset('CVL_raw', 'CVL_Images')
    
    image_dict, pairs_dict = preprocess_all_images('CVL_Images', 'Preprocessed')
    print(f"Images preprocessed {[len(v) for v in image_dict.values()]}")
    

    # Optionally prefix IAM writer IDs to avoid ID collision
    iam_dict_prefixed = {f'cvl_{k}': v for k, v in image_dict.items()}
    
    # Merge both datasets
    image_dict.update(iam_dict_prefixed)
    """
    
    # Step 2: Generate image pairs
    print("Generating image pairs...")
    pairs, labels = generate_pairs(pairs_dict, images_per_writer)
    print(f"Generated {len(pairs)} pairs")
    
    #Showing about 10 random pairs to inspect preprocessing results
    visualize_and_show_pair(pairs, labels, num_samples=10)
    
    # Step 3: Train/test split
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        pairs, labels, test_size=0.2, random_state=42)
    
    #In production version we shall comment line below as it only limits to 1000 data to make it train faster
    #train_pairs, train_labels = train_pairs[:1000], train_labels[:1000]
    #test_pairs, test_labels = test_pairs[:300], test_labels[:300]

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
    
    
    # Step 7: Training the model
    train_losses, train_accuracies = train_siamese(model, train_loader, criterion, optimizer, device, num_epochs=epochsAllowed)
    
    # Plotting training performance
    plot_training_curves(train_losses, train_accuracies)
    
    # Loading the best model for evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    print("Loaded best model for evaluation.")
    
    # Evaluate
    evaluate_model(model, test_loader, device, export_csv_path="predictions.csv")


if __name__ == "__main__":
    main()

