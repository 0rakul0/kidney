import os
import cv2
import torch
import numpy as np
import torchvision
import torch.nn as nn
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_FOLDER = "dataset_loader"

IDENTIFICADA_IMG = "identificada/image"
IDENTIFICADA_MASK = "identificada/mask"

NAO_IDENTIFICADA = "nao_identificada"

MODEL_PATH = "models/best_model.pth"

IMG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.90
PIXEL_THRESHOLD = 0.7

# criar pastas
os.makedirs(IDENTIFICADA_IMG, exist_ok=True)
os.makedirs(IDENTIFICADA_MASK, exist_ok=True)
os.makedirs(NAO_IDENTIFICADA, exist_ok=True)

# =========================
# LOAD MODEL
# =========================

model = torchvision.models.segmentation.deeplabv3_resnet50(
    weights=None,
    aux_loss=True
)

model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

# =========================
# PREPROCESSAMENTO
# =========================

def remover_frame_preto(img):

    coords = np.column_stack(np.where(img > 10))

    if len(coords) == 0:
        return img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return img[y_min:y_max, x_min:x_max]


def limpar_ultrassom(img):

    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img = clahe.apply(img)

    return img


def normalizar(img):

    img = img.astype("float32")

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img


def preparar_tensor(img):

    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

    img = normalizar(img)

    img = np.stack([img,img,img],axis=0)

    tensor = torch.tensor(img,dtype=torch.float32).unsqueeze(0)

    return tensor


# =========================
# PÓS PROCESSAMENTO
# =========================

def largest_component(mask):

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return (labels == largest).astype(np.uint8)


def limpar_mascara(mask):

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# =========================
# PROCESSAMENTO
# =========================

files = os.listdir(INPUT_FOLDER)

for file in tqdm(files):

    path = os.path.join(INPUT_FOLDER,file)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    # remover frame
    img = remover_frame_preto(img)

    # limpar ultrassom
    img = limpar_ultrassom(img)

    tensor = preparar_tensor(img).to(DEVICE)

    with torch.no_grad():

        pred = model(tensor)["out"]

    prob = torch.sigmoid(pred).cpu().numpy()[0,0]

    mask = (prob > PIXEL_THRESHOLD).astype(np.uint8)

    mask = largest_component(mask)

    mask = limpar_mascara(mask)

    if mask.sum() == 0:

        cv2.imwrite(os.path.join(NAO_IDENTIFICADA,file), img)

        continue

    confidence = prob[mask == 1].mean()

    if confidence >= CONFIDENCE_THRESHOLD:

        mask_save = (mask*255).astype(np.uint8)

        img_save = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        cv2.imwrite(
            os.path.join(IDENTIFICADA_IMG,file),
            img_save
        )

        cv2.imwrite(
            os.path.join(IDENTIFICADA_MASK,file),
            mask_save
        )

    else:

        cv2.imwrite(
            os.path.join(NAO_IDENTIFICADA,file),
            img
        )

print("Processamento finalizado")