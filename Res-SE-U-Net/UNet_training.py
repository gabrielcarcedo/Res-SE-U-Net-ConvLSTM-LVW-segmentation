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
from scipy.spatial import distance
import re
import csv

# Definir dataset personalizado
class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Cargar las imágenes y máscaras con PIL
        image = Image.open(self.image_paths[idx]).convert('L')  # Convertir a escala de grises
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Convertir a tensor y normalizar la imagen
        if self.transform:
            image = self.transform(image)  # Convertir la imagen a tensor aquí
            mask = self.transform(mask)  # Convertir la máscara a tensor aquí

        return image, mask

dataset_path = 'Dataset_avances/resized_images/'

# Rutas de las imágenes y las máscaras
path_training_images = os.path.join(dataset_path, 'train_images_aumentado')
path_training_masks = os.path.join(dataset_path, 'train_masks_aumentado')
path_test_images = os.path.join(dataset_path, 'test_images')
path_test_masks = os.path.join(dataset_path, 'test_masks')

# Cargar las rutas de las imágenes y las máscaras
train_images = [os.path.join(path_training_images, i) for i in os.listdir(path_training_images)]
train_masks = [os.path.join(path_training_masks, i) for i in os.listdir(path_training_masks)]
test_images = [os.path.join(path_test_images, i) for i in os.listdir(path_test_images)]
test_masks = [os.path.join(path_test_masks, i) for i in os.listdir(path_test_masks)]

# Verificación de existencia de archivos
for image_path, mask_path in zip(train_images, train_masks):
    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise Exception(f"Archivo no encontrado: {image_path}, {mask_path}")

train_images.sort()
train_masks.sort()
test_images.sort()
test_masks.sort()

print('------- LISTAS DE IMAGES/MASKS -------')
print(len(train_images))
print(len(train_masks))
print(len(test_images))
print(len(test_masks))


# Cargar un modelo U-Net preentrenado
model = smp.Unet(
    encoder_name="resnet18",  # Se puede elegir otro encoder preentrenado
    encoder_weights="imagenet",  # Cargar pesos preentrenados de ImageNet
    decoder_attention_type='scse',
    in_channels=1,  # Número de canales de entrada
    classes=1,  # Número de clases en la segmentación (en este caso segmentación binaria)
).to('cuda')

# Definir el optimizador y la función de pérdida
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()  # Para tareas de segmentación binaria

# Definir un transformador para redimensionar las imágenes
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Ajustar a tamaño divisible por 32
    transforms.ToTensor(),
])

# Crear los datasets
train_dataset = CustomDataset(train_images, train_masks, transform=transform)
test_dataset = CustomDataset(test_images, test_masks, transform=transform)

# Crear los dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Función para calcular IoU
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()  # Binarizar la predicción
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = intersection / union if union != 0 else 0  # Evitar división por cero
    return iou

# Función para guardar una imagen
def save_image(tensor, filename):
    # Asegurando que el tensor esté en el rango [0, 1]
    tensor = tensor.squeeze(0).cpu().detach()  # Eliminar el batch dimension
    tensor = torch.clamp(tensor, 0, 1)  # Limitar a rango [0, 1]

    # Convertir el tensor a imagen
    transform_to_pil = transforms.ToPILImage()
    pil_image = transform_to_pil(tensor)  # Convertir tensor a imagen PIL

    # Guardar la imagen
    pil_image.save(filename)

model_description = 'UNet_ResNet18_SE'
fecha = '27_abril_2025'

# Crear carpeta para guardar los checkpoints si no existe
path_root = dataset_path + model_description + '/'
checkpoint_dir = path_root + 'checkpoints_crossval/'
os.makedirs(checkpoint_dir, exist_ok=True)

# Crear una carpeta para guardar las imágenes de salida
output_dir = os.path.join(path_root, 'test_outputs/')
os.makedirs(output_dir, exist_ok=True)

# Crear una carpeta para guardar las gráficas de pérdidas y IoU en entrenamiento
graph_dir = os.path.join(path_root, 'gráficas_loss_val')
os.makedirs(graph_dir, exist_ok=True)

# File path for the CSV file
csv_file_path = f'{path_root}results_{model_description}_{fecha}.csv'
data_csv = [['K', 'Epoch', 'Loss', 'IoU', 'Precision', 'Recall', 'F1']]

best_model_iou, best_k, best_ckpt_id, estacionamiento = 0.0, 0, '', 0

for k in range(5):
    k+=1
    print(f'-------------------------FOLD {k}-------------------------')

    # Cargar un modelo U-Net preentrenado
    model = smp.Unet(
        encoder_name="resnet18",  # Se puede elegir otro encoder preentrenado
        encoder_weights="imagenet",  # Cargar pesos preentrenados de ImageNet
        decoder_attention_type='scse',
        in_channels=1,  # Número de canales de entrada
        classes=1,  # Número de clases en la segmentación (en este caso segmentación binaria)
    ).to('cuda')

    # Definir el optimizador y la función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()  # Para tareas de segmentación binaria

    # Crear carpeta para guardar los checkpoints si no existe
    ckpt_id = 'K_' + str(k) + '_' + fecha

    # Inicializar variables para llevar el control del mejor modelo
    best_iou = 0.0
    best_epoch = 0

    # Ciclo de entrenamiento
    num_epochs = 200
    losses = []
    iou_values = []
    precision_values = []
    recall_values = []
    f1_values = []
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

        # Calcular pérdida promedio por epoch
        average_loss = running_loss / len(train_loader)
        losses.append(average_loss)
        print(f"\nFold {k}, Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

        # Evaluar en el conjunto de validación o test
        model.eval()
        running_iou = 0.0
        running_precision = 0.0
        running_recall = 0.0
        running_f1 = 0.0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to('cuda'), masks.to('cuda')

                outputs = model(images)

                # Calcular IoU para evaluar
                iou = compute_iou(outputs, masks)
                running_iou += iou.item()

                preds = (outputs > 0.5).float()
                preds_np = preds.detach().cpu().numpy().astype(np.uint8).flatten()
                masks_np = masks.detach().cpu().numpy().astype(np.uint8).flatten()

                precision = precision_score(masks_np, preds_np)
                recall = recall_score(masks_np, preds_np)
                f1 = f1_score(masks_np, preds_np)

                running_precision += precision
                running_recall += recall
                running_f1 += f1

        average_iou = running_iou / len(test_loader)
        iou_values.append(average_iou)
        print(f"Validation IoU: {average_iou:.4f}")

        average_precision = running_precision / len(test_loader)
        precision_values.append(average_precision)
        print(f"Validation Precision: {average_precision:.4f}")

        average_recall = running_recall / len(test_loader)
        recall_values.append(average_recall)
        print(f"Validation Recall: {average_recall:.4f}")

        average_f1 = running_f1 / len(test_loader)
        f1_values.append(average_f1)
        print(f"Validation F1: {average_f1:.4f}")

        data_csv.append([k, epoch+1, average_loss, average_iou, average_precision, average_recall, average_f1])

        # Guardar checkpoint si mejora el IoU
        if average_iou > best_iou:
            best_iou = average_iou
            best_epoch = epoch + 1

            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{ckpt_id}.pth")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, checkpoint_path)

            print(f"Checkpoint guardado en Fold {k} en epoch {best_epoch} con IoU: {best_iou:.4f}")

        if best_iou > best_model_iou:
            best_model_iou = best_iou
            best_k = k
            best_ckpt_id = ckpt_id
            estacionamiento = 0
          
        estacionamiento +=1

        if estacionamiento == 40:
            break

    # Graficar la pérdida a lo largo de las épocas
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='--', label='training losses', color='#e63a22', lw=3)
    plt.plot(iou_values, marker='x', linestyle='-.', label='IoU validation', color='#60ca23', lw=3)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Loss & IoU Validation Over Epochs')
    plt.grid(True, ls='--', alpha=0.5, color='lightgray')
    plt.legend()
    plt.savefig(f'{graph_dir}/loss_iou_epochs_{ckpt_id}.png')
    plt.show()

with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data_csv)
