import re
import random
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('TkAgg')


os.makedirs('Training_images/Paper', exist_ok=True)
os.makedirs('Training_images/Tablet', exist_ok=True)


# CONVOLUTIONAL BRANCH
class ConvolutionalBranch(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # * Can experiment with different pooling strategies

        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding='same')  # *
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # *

        self.conv3 = nn.Conv2d(64, 96, (3, 3), padding='same')  # *
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))  # *

        self.conv4 = nn.Conv2d(96, 128, (3, 3), padding='same')  # *
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))  # *

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # * Can replace with flatten or other pooling
        self.dense_embedding = nn.Linear(128, 512)  # * Embedding size and activation can be tuned

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # * Can adjust number of filters or kernel size
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))  # *
        x = self.pool2(x)  # *

        x = torch.relu(self.conv3(x))  # *
        x = self.pool3(x)  # *

        x = torch.relu(self.conv4(x))  # *
        x = self.pool4(x)  # *

        x = self.gap(x).view(x.size(0), -1)
        x = torch.relu(self.dense_embedding(x))  # * Embedding size and activation can be tuned

        return x


# SIMPLE MUTUAL ATTENTION MECHANISM
def mutual_attention(features_main, features_inverse):
    weighted = features_main * features_inverse  # * Can replace with more advanced attention mechanisms
    fused = weighted + features_inverse
    return fused


# MAIN MODEL BUILDER
class HandwritingVerificationModel(nn.Module):
    def __init__(self, input_shape=(1, 155, 220)):  # assuming all images were resized to 220x155
        super().__init__()
        self.shared_feature_extractor = ConvolutionalBranch(input_channels=input_shape[0])

        # * Pairing strategies can be refined
        self.fc1 = nn.Linear(512 * 2, 128)  # * Can tune size, activation
        self.fc2 = nn.Linear(512 * 2, 128)  # *
        self.fc3 = nn.Linear(512 * 2, 128)  # *
        self.fc4 = nn.Linear(512 * 2, 128)  # *

        self.out1 = nn.Linear(128, 1)  # * Activation or binary decision structure can be tuned
        self.out2 = nn.Linear(128, 1)
        self.out3 = nn.Linear(128, 1)
        self.out4 = nn.Linear(128, 1)

    def forward(self, input_text_ref, input_text_query, input_text_ref_inv, input_text_query_inv):
        features_ref = self.shared_feature_extractor(input_text_ref)
        features_query = self.shared_feature_extractor(input_text_query)
        features_ref_inv = self.shared_feature_extractor(input_text_ref_inv)
        features_query_inv = self.shared_feature_extractor(input_text_query_inv)

        fused_ref = mutual_attention(features_ref, features_ref_inv)  # * Can adjust fusion logic
        fused_query = mutual_attention(features_query, features_query_inv)  # *

        pair_1 = torch.cat([fused_ref, features_query], dim=1)
        pair_2 = torch.cat([features_ref, features_query], dim=1)
        pair_3 = torch.cat([features_ref, fused_query], dim=1)
        pair_4 = torch.cat([fused_ref, fused_query], dim=1)

        fc1 = torch.relu(self.fc1(pair_1))  # * Can tune size, activation
        fc2 = torch.relu(self.fc2(pair_2))  # *
        fc3 = torch.relu(self.fc3(pair_3))  # *
        fc4 = torch.relu(self.fc4(pair_4))  # *

        out1 = torch.sigmoid(self.out1(fc1))  # * Activation or binary decision structure can be tuned
        out2 = torch.sigmoid(self.out2(fc2))
        out3 = torch.sigmoid(self.out3(fc3))
        out4 = torch.sigmoid(self.out4(fc4))

        final_output = torch.mean(torch.stack([out1, out2, out3, out4], dim=0), dim=0)  # * Can replace with voting or trainable fusion
        return final_output.squeeze(1)

# CREATE PAIRS (Same Writer and Different Writers)
def build_writer_dict_from_list(image_paths):
    writer_dict = {}
    id_pattern = re.compile(r'(id_[A-Za-z0-9]+)')
    for img_path in image_paths:
        match = id_pattern.search(os.path.basename(img_path))
        if match:
            writer_id = match.group(1)
            writer_dict.setdefault(writer_id, []).append(img_path)
    return writer_dict

def generate_pairs(num_pairs, writer_dict, writer_ids):
    pairs = []

    for _ in range(num_pairs):
        if random.random() < 0.5:
            # Positive pair (same writer)
            writer = random.choice(writer_ids)
            if len(writer_dict[writer]) >= 2:
                img1, img2 = random.sample(writer_dict[writer], 2)
                label = 1
            else:
                continue

        else:
            # Negative pair (different writers)
            w1, w2 = random.sample(writer_ids, 2)
            img1 = random.choice(writer_dict[w1])
            img2 = random.choice(writer_dict[w2])
            label = 0

        pairs.append((img1, img2, label))
    return pairs


# DATA PIPELINE
class HandwritingPairsDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        arr_ref = np.array(Image.open(img1_path).resize((220, 155))).astype('float32') / 255.0
        arr_ref = np.expand_dims(arr_ref, axis=0)
        arr_ref_inv = 1.0 - arr_ref

        arr_query = np.array(Image.open(img2_path).resize((220, 155))).astype('float32') / 255.0
        arr_query = np.expand_dims(arr_query, axis=0)
        arr_query_inv = 1.0 - arr_query

        return (
            torch.tensor(arr_ref, dtype=torch.float32),
            torch.tensor(arr_query, dtype=torch.float32),
            torch.tensor(arr_ref_inv, dtype=torch.float32),
            torch.tensor(arr_query_inv, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


# --------- MAIN PIPELINE ---------
def main():
    """
    Assumes labelled images are already available in 'Training_images/Paper' and
    'Training_images/Tablet' folders, with filenames containing writer IDs.
    No additional preprocessing or image downloading is performed in this script.
    """

    paper_groups = sorted([os.path.join('Training_images/Paper', x) for x in os.listdir('Training_images/Paper')])
    tab_groups = sorted([os.path.join('Training_images/Tablet', x) for x in os.listdir('Training_images/Tablet')])
    full_train_data = paper_groups[:151] + tab_groups[:71]

    writer_dict = build_writer_dict_from_list(full_train_data)
    writer_ids = list(writer_dict.keys())
    pairs = generate_pairs(num_pairs=10, writer_dict=writer_dict, writer_ids=writer_ids)

    # # Train-test Split 90:10
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.10, random_state=42, shuffle=True)
    train_dataset = HandwritingPairsDataset(train_pairs)
    test_dataset = HandwritingPairsDataset(test_pairs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HandwritingVerificationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    best_loss = float('inf')
    patience = 5
    no_improve_epochs = 0
    epochs = 20  # * Can be adjusted

    for epoch in range(epochs):
        # Train the model
        model.train()
        train_loss = 0.0
        for ref, query, ref_inv, query_inv, labels in train_loader:
            ref, query, ref_inv, query_inv, labels = (ref.to(device), query.to(device), ref_inv.to(device),
                                                      query_inv.to(device), labels.to(device))
            optimizer.zero_grad()
            outputs = model(ref, query, ref_inv, query_inv)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        y_true, y_pred_probs = [], []
        with torch.no_grad():
            for ref, query, ref_inv, query_inv, labels in test_loader:
                ref, query, ref_inv, query_inv, labels = ref.to(device), query.to(device), ref_inv.to(device), query_inv.to(device), labels.to(device)
                outputs = model(ref, query, ref_inv, query_inv)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred_probs.extend(outputs.cpu().numpy())
        val_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        # ModelCheckpoint
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print('EarlyStopping(patience=5)')
                break

    # Model Evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    y_true, y_pred_probs = [], []
    with torch.no_grad():
        for ref, query, ref_inv, query_inv, labels in test_loader:
            ref, query, ref_inv, query_inv = ref.to(device), query.to(device), ref_inv.to(device), query_inv.to(device)
            outputs = model(ref, query, ref_inv, query_inv)

            for i in range(len(outputs)):
                prob = outputs[i].item()
                pred_label = 1 if prob > 0.5 else 0
                true_label = int(labels[i].item())
                print(
                    f'Prediction: {'Same Writer' if pred_label == 1 else 'Different Writers'} | Confidence: '
                    f'{prob * 100:.2f}% | True Label: {true_label}')
                
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(outputs.cpu().numpy())
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int).flatten()
    y_true = np.array(y_true).flatten()

    # Metrics Calculation
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)

    print(f'Accuracy:  {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall:    {recall:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    print('Confusion Matrix:\n', confusion_mat)


""""
    # Confusion Matrix Plot- optional
    plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
"""


if __name__ == '__main__':
    main()
