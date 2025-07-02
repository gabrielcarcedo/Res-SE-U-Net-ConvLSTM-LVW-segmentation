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
from skimage import metrics
from scipy.spatial import distance
from sklearn.model_selection import KFold
import re
import csv

class ConvLSTMDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.inputs = input_sequences
        self.targets = target_sequences

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(combined_conv, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device)
        )

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers

        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell = ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size)
            cell_list.append(cell)

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        # input_tensor: (B, T, C, H, W)
        b, t, c, h, w = input_tensor.size()
        hidden_states = []

        for layer_idx in range(self.num_layers):
            h_cur, c_cur = self.cell_list[layer_idx].init_hidden(b, (h, w))
            hidden_states.append((h_cur, c_cur))

        layer_output = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_states[layer_idx]
            output_inner = []

            for time_step in range(t):
                h, c = self.cell_list[layer_idx](layer_output[:, time_step], (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  # (B, T, C, H, W)

        return layer_output

class ConvLSTMNetManyToMany(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, kernel_size=3, num_layers=2):
        super(ConvLSTMNetManyToMany, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.encoder_pool = nn.MaxPool2d(2)

        self.convlstm = ConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers
        )

        # Decoder: ahora va a recibir canales dobles por la concatenación
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        B, T, C, H, W = input_tensor.size()
        x = input_tensor.view(B * T, C, H, W)

        # Encoder
        encoder_features = self.encoder_conv(x)      # (B*T, 32, H, W)
        x_pooled = self.encoder_pool(encoder_features)  # (B*T, 32, H/2, W/2)

        # Reshape para pasar por ConvLSTM
        _, C_enc, H_enc, W_enc = x_pooled.shape
        x_pooled = x_pooled.view(B, T, C_enc, H_enc, W_enc)
        lstm_out = self.convlstm(x_pooled)  # (B, T, 32, H/2, W/2)

        # Decoder con skip-connection
        out = []
        encoder_features = encoder_features.view(B, T, -1, H, W)  # reshape back
        for t in range(T):
            # Upsample + concat skip-connection
            lstm_frame = lstm_out[:, t]                           # (B, 32, H/2, W/2)
            skip = encoder_features[:, t]                        # (B, 32, H, W)
            lstm_frame_up = F.interpolate(lstm_frame, scale_factor=2, mode='bilinear', align_corners=False)
            concat = torch.cat([lstm_frame_up, skip], dim=1)     # (B, 64, H, W)
            decoded = self.decoder(concat)                       # (B, 1, H, W)
            out.append(decoded)

        return torch.stack(out, dim=1)  # (B, T, 1, H, W)
  
def generate_sequences_from_folder(folder_path, sequence_length=4, output_size=(256, 256)):
    segmentation_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))
    transform_mask = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),  # (1, H, W)
    ])
    sequences = []
    i = 0
    while i + sequence_length <= len(segmentation_paths):
        sequence = []
        for j in range(sequence_length):
            mask_path = segmentation_paths[i + j]
            mask = Image.open(mask_path).convert('L')
            mask_tensor = transform_mask(mask)
            sequence.append(mask_tensor)
        sequence_tensor = torch.stack(sequence, dim=0)  # (T, 1, H, W)
        sequences.append(sequence_tensor)
        i += sequence_length  # sin solapamiento
    return sequences

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def generate_input_target_sequences(input_folder, mask_folder, sequence_length):
    #input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')], key=natural_sort_key)
    input_files = [f for f in os.listdir(input_folder) if f.endswith('.png')] 
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
    #mask_files.sort()

    for i in range(len(input_files)):
        if input_files[i].split('Segmentation_')[-1] != mask_files[i]:
            print(f"Error: {input_files[i]} != {mask_files[i]}")

    input_sequences = []
    mask_sequences = []

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for i in range(0, len(input_files) - sequence_length + 1, sequence_length):  # ¡avance sin solapamiento!
        input_seq = []
        mask_seq = []

        for j in range(sequence_length):
            input_path = os.path.join(input_folder, input_files[i + j])
            input_image = Image.open(input_path).convert('L')
            input_tensor = transform(input_image)

            mask_path = os.path.join(mask_folder, mask_files[i + j])
            mask_image = Image.open(mask_path).convert('L')
            mask_tensor = transform(mask_image)

            input_seq.append(input_tensor)
            mask_seq.append(mask_tensor)

        input_sequences.append(torch.stack(input_seq))  # (T, 1, 256, 256)
        mask_sequences.append(torch.stack(mask_seq))

    return input_sequences, mask_sequences
  
def sequence_loss(pred_seq, target_seq):
    # pred_seq, target_seq: (B, T, 1, H, W)
    loss = 0
    for t in range(pred_seq.size(1)):
        loss += nn.BCEWithLogitsLoss()(pred_seq[:, t], target_seq[:, t])
    return loss / pred_seq.size(1)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Aplicamos sigmoid a los logits
        inputs = inputs.reshape(-1)  # Usamos reshape en lugar de view
        targets = targets.reshape(-1)  # Usamos reshape en lugar de view
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1.0):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, inputs, targets):
        bce = self.bce(inputs, targets)
        dice = self.dice(inputs, targets)
        return self.alpha * bce + (1 - self.alpha) * dice

def sequence_temporal_loss(pred_seq, target_seq, lambda_temp=0.1, combo_alpha=0.75):
    """
    Combina ComboLoss (BCE + Dice) frame a frame con una penalización de suavidad temporal.
    """
    combo = ComboLoss(alpha=combo_alpha)
    combo_loss = 0
    for t in range(pred_seq.size(1)):
        combo_loss += combo(pred_seq[:, t], target_seq[:, t])

    combo_loss = combo_loss / pred_seq.size(1)

    # Penalización temporal de suavidad
    temporal_diff = (torch.sigmoid(pred_seq[:, 1:]) - torch.sigmoid(pred_seq[:, :-1])) ** 2
    temporal_smoothness = temporal_diff.mean()

    total_loss = combo_loss + temporal_smoothness
    return total_loss

def iou_metric(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()
    intersection = (pred_bin * target_bin).sum(dim=(2, 3, 4))
    union = (pred_bin + target_bin - pred_bin * target_bin).sum(dim=(2, 3, 4))
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

sequence_length = 4
model_description = 'model_description'
ckpt_id = 'ckpt_id'

train_input_dir = os.path.join(dataset_path, f'train_outputs/{ckpt_id}')
train_mask_dir = os.path.join(dataset_path, f'train_masks')
test_input_dir = os.path.join(dataset_path, f'test_outputs/{ckpt_id}')
test_mask_dir = os.path.join(dataset_path, f'test_masks')

train_inputs, train_targets = generate_input_target_sequences(train_input_dir, train_mask_dir, sequence_length)
test_inputs, test_targets = generate_input_target_sequences(test_input_dir, test_mask_dir, sequence_length)

print(f"Secuencias de entrenamiento: {len(train_inputs)}")
print(f"Targets de entrenamiento: {len(train_targets)}")
print(f"Secuencias de prueba: {len(test_inputs)}")
print(f"Targets de prueba: {len(test_targets)}")
print(f"Shape de una secuencia de input: {train_inputs[0].shape}")
print(f"Shape de su máscara: {train_targets[0].shape}")

# Crear carpeta para guardar los checkpoints si no existe
path_root = dataset_path + model_description + '/'
checkpoint_dir = path_root + 'checkpoints_crossval/'
os.makedirs(checkpoint_dir, exist_ok=True)

# Crear una carpeta para guardar las gráficas de pérdidas y IoU en entrenamiento
graph_dir = os.path.join(path_root, 'gráficas_loss_val')
os.makedirs(graph_dir, exist_ok=True)

# File path for the CSV file
csv_file_path = f'{path_root}results_{model_description}_{fecha}.csv'

# Definir K folds
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
folds = list(kf.split(train_inputs))

data_csv = [['K', 'Epoch', 'Loss', 'IoU']]

# Entrenamiento con validación cruzada
for fold_idx, (train_idx, val_idx) in enumerate(folds):
    print(f"\n------------ FOLD {fold_idx + 1} / {K} -----------")

    fold_train_inputs = [train_inputs[i] for i in train_idx]
    fold_train_targets = [train_targets[i] for i in train_idx]
    fold_val_inputs = [train_inputs[i] for i in val_idx]
    fold_val_targets = [train_targets[i] for i in val_idx]

    train_dataset = ConvLSTMDataset(fold_train_inputs, fold_train_targets)
    val_dataset = ConvLSTMDataset(fold_val_inputs, fold_val_targets)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Crear nuevo modelo por fold
    model = ConvLSTMNetManyToMany(input_channels=1, hidden_channels=32)
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_iou = 0.0
    estacionamiento = 0
    losses, ious = [], []

    for epoch in range(100):  # num_epochs
        model.train()
        train_loss = 0
        for input_seq, target_seq in train_loader:
            input_seq = input_seq.to('cuda')
            target_seq = target_seq.to('cuda')

            output_seq = model(input_seq)
            loss = sequence_temporal_loss(output_seq, target_seq)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validación
        model.eval()
        total_iou = 0
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq = input_seq.to('cuda')
                target_seq = target_seq.to('cuda')
                pred_seq = model(input_seq)
                iou = iou_metric(pred_seq, target_seq)
                total_iou += iou.item()

        avg_iou = total_iou / len(val_loader)
        avg_loss = train_loss / len(train_loader)

        print(f"Fold: {fold_idx + 1} - Epoch: {epoch+1}/100 - Train Loss: {avg_loss:.4f} - Val IoU: {avg_iou:.4f}")
        losses.append(avg_loss)
        ious.append(avg_iou)
        data_csv.append([fold_idx + 1, epoch + 1, avg_loss, avg_iou])

        # Guardar el mejor modelo
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold_idx + 1}.pth")
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, checkpoint_path)
            print(f"Checkpoint guardado en epoch {best_epoch} con IoU: {best_iou:.4f}")
            estacionamiento = 0
        else:
            estacionamiento += 1

        if estacionamiento >= 20:
            break

    # Gráfica por fold
    plt.figure(figsize=(10, 5))
    plt.plot(losses, marker='o', linestyle='--', label='Training losses', color='#e63a22', lw=3)
    plt.plot(ious, marker='x', linestyle='-.', label='IoU validation', color='#60ca23', lw=3)
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(f'Fold {fold_idx + 1}: Training Loss & IoU Validation')
    plt.grid(True, ls='--', alpha=0.5, color='lightgray')
    plt.legend()
    plt.savefig(f'{graph_dir}/loss_iou_epochs_K_{fold_idx + 1}_{ckpt_id}.png')
    plt.close()

# Guardar CSV
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data_csv)
