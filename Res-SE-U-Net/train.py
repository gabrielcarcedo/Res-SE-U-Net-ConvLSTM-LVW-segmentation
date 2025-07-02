import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
import csv

# Dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Configuración de rutas y dataset
dataset_path = 'dataset_path'
path_training_images = os.path.join(dataset_path, 'path_training_images')
path_training_masks = os.path.join(dataset_path, 'path_training_masks')

train_images = sorted([os.path.join(path_training_images, i) for i in os.listdir(path_training_images)])
train_masks = sorted([os.path.join(path_training_masks, i) for i in os.listdir(path_training_masks)])

print('------- LISTAS DE IMAGES/MASKS -------')
print(len(train_images))
print(len(train_masks))

# Modelo y parámetros
model_description = 'model_description'
fecha = 'fecha'

path_root = os.path.join(dataset_path, model_description)
checkpoint_dir = os.path.join(path_root, 'checkpoints_crossval')
graph_dir = os.path.join(path_root, 'gráficas_loss_val')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

csv_file_path = f'{path_root}/results_{model_description}_{fecha}.csv'
data_csv = [['K', 'Epoch', 'Loss', 'IoU', 'Precision', 'Recall', 'F1']]

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Métricas
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / union if union != 0 else 0

def save_image(tensor, filename):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = torch.clamp(tensor, 0, 1)
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(tensor)
    pil_image.save(filename)

# Validación cruzada K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
best_model_iou = 0.0
best_k = 0
best_ckpt_id = ''
estacionamiento = 0

for k, (train_index, val_index) in enumerate(kf.split(train_images)):
    print(f'-------------------------FOLD {k+1}-------------------------')

    train_imgs_fold = [train_images[i] for i in train_index]
    train_masks_fold = [train_masks[i] for i in train_index]
    val_imgs_fold   = [train_images[i] for i in val_index]
    val_masks_fold  = [train_masks[i] for i in val_index]

    train_dataset = CustomDataset(train_imgs_fold, train_masks_fold, transform=transform)
    val_dataset   = CustomDataset(val_imgs_fold, val_masks_fold, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        decoder_attention_type='scse',
        in_channels=1,
        classes=1,
    ).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    ckpt_id = f"K_{k+1}_{fecha}"
    best_iou = 0.0
    best_epoch = 0

    num_epochs = 200
    losses, iou_values, precision_values, recall_values, f1_values = [], [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to('cuda'), masks.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"\nFold {k+1}, Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Evaluación
        model.eval()
        running_iou = running_precision = running_recall = running_f1 = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to('cuda'), masks.to('cuda')
                outputs = model(images)

                iou = compute_iou(outputs, masks)
                running_iou += iou.item()

                preds = (outputs > 0.5).float()
                preds_np = preds.cpu().numpy().astype(np.uint8).flatten()
                masks_np = masks.cpu().numpy().astype(np.uint8).flatten()

                precision = precision_score(masks_np, preds_np, zero_division=0)
                recall = recall_score(masks_np, preds_np, zero_division=0)
                f1 = f1_score(masks_np, preds_np, zero_division=0)

                running_precision += precision
                running_recall += recall
                running_f1 += f1

        avg_iou = running_iou / len(val_loader)
        avg_precision = running_precision / len(val_loader)
        avg_recall = running_recall / len(val_loader)
        avg_f1 = running_f1 / len(val_loader)

        iou_values.append(avg_iou)
        precision_values.append(avg_precision)
        recall_values.append(avg_recall)
        f1_values.append(avg_f1)

        print(f"Validation IoU: {avg_iou:.4f}")
        print(f"Validation Precision: {avg_precision:.4f}")
        print(f"Validation Recall: {avg_recall:.4f}")
        print(f"Validation F1: {avg_f1:.4f}")

        data_csv.append([k+1, epoch+1, avg_loss, avg_iou, avg_precision, avg_recall, avg_f1])

        # Guardar checkpoint si mejora
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, os.path.join(checkpoint_dir, f"best_model_{ckpt_id}.pth"))
            print(f"Checkpoint guardado en Fold {k+1}, Epoch {best_epoch}, IoU: {best_iou:.4f}")
            estacionamiento = 0

        if best_iou > best_model_iou:
            best_model_iou = best_iou
            best_k = k+1
            best_ckpt_id = ckpt_id

        estacionamiento += 1
        if estacionamiento == 40:
            break

    # Gráfica por fold
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='--', label='Training Loss', color='#e63a22', lw=3)
    plt.plot(iou_values, marker='x', linestyle='-.', label='Validation IoU', color='#60ca23', lw=3)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(f'Training Loss & IoU - Fold {k+1}')
    plt.grid(True, ls='--', alpha=0.5, color='lightgray')
    plt.legend()
    plt.savefig(os.path.join(graph_dir, f'loss_iou_epochs_{ckpt_id}.png'))
    plt.close()

# Guardar resultados finales en CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data_csv)

