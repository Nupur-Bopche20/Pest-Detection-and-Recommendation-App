import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# --- Step 1: Configuration and Data Preparation (with Enhanced Augmentation) ---

def setup_data_loaders(data_dir, batch_size=32, data_transforms=None):
    """
    Prepares and returns the training and validation data loaders.
    """
    if data_transforms is None:
        raise ValueError("data_transforms must be provided")

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    print("Loading datasets...")
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }

    print("Data loaders created successfully.")
    return dataloaders, image_datasets

# --- Step 2: Hybrid Model Architecture (Unchanged) ---

class HybridPestNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridPestNet, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.custom_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        vit_features = self.vit(x)
        resnet_features = self.resnet(x)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        cnn_features = self.custom_cnn(x)
        combined_features = torch.cat((vit_features, resnet_features, cnn_features), dim=1)
        output = self.classifier(combined_features)
        return output

# --- Step 3: Training the Model (FIXED) ---

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=50, patience=7):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    epochs_no_improve = 0
    best_val_loss = float('inf')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        # --- FIX: Added flush=True to ensure this prints immediately ---
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-'*10, flush=True)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=(phase == 'train')):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                scheduler.step(epoch_loss)

            # --- FIX: Added flush=True to ensure this prints immediately ---
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', flush=True)

            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f'\nEarly stopping triggered after {patience} epochs with no improvement.')
                    model.load_state_dict(best_model_wts)
                    return model, history

    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# --- Step 4: Model Evaluation (Unchanged) ---
def evaluate_model(model, data_dir, class_names, device):
    test_dir = os.path.join(data_dir, 'test')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\nClassification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cm, index=class_names, columns=class_names), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'), plt.ylabel('Actual')
    plt.show()

# --- Step 5: Prediction on a Single Image (Removed - now in a separate cell) ---

# --- Main Execution Block ---
if __name__ == '__main__':
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks/pest_dataset_split'

    NUM_EPOCHS = 50
    PATIENCE = 7
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-4

    if not os.path.isdir(DATA_DIR) or not os.path.isdir(os.path.join(DATA_DIR, 'train')):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please update the 'DATA_DIR' variable in the script.")
    else:
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.TrivialAugmentWide(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(imagenet_mean, imagenet_std)
            ]),
        }


        dataloaders, image_datasets = setup_data_loaders(DATA_DIR, BATCH_SIZE, data_transforms)
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        print(f"Found {num_classes} classes: {', '.join(class_names)}")

        model = HybridPestNet(num_classes=num_classes).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

        trained_model, history = train_model(model, dataloaders, criterion, optimizer, scheduler, DEVICE, num_epochs=NUM_EPOCHS, patience=PATIENCE)

        model_save_path = 'pest_detection_hybrid_model_optimized.pth'
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"\nModel saved to: {model_save_path}")

        print("\n--- Evaluating on Test Set ---")
        evaluate_model(trained_model, DATA_DIR, class_names, DEVICE)