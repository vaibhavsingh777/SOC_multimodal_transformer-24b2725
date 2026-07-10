

# ==========================================================
# CLIP Training Script
# Task 6 - VLM from Scratch
# ==========================================================

import os
import csv
import time
import random
import numpy as np
import pandas as pd

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)

import torchvision.transforms as transforms

from dataset import (
    Tokenizer,
    Flickr8kDataset,
    collate_fn,
)

from clip_model import (
    ViTEncoder,
    TextEncoder,
    CLIPStyleModel,
)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

print("Working directory:", os.getcwd())
# ==========================================================
# Configuration
# ==========================================================

CONFIG = {

    "image_size":64,
    "max_text_len":32,

    "embed_dim":192,
    "projection_dim":128,
    "patch_size":8,
    "vit_depth":4,
    "text_depth":4,
    "n_head":6,
    "dropout":0.1,

    "batch_size":128,
    "epochs":20,

    "lr":5e-4,
    "weight_decay":0.05,

    "warmup_steps":500,
    "total_steps":10000,

    "grad_clip":1.0,

    "val_every":200,

    "seed":42,
}

# ==========================================================
# Reproducibility
# ==========================================================

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

torch.manual_seed(CONFIG["seed"])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ==========================================================
# Load Captions
# ==========================================================

captions_df = pd.read_csv(
    "data/Flickr8k/captions.txt"
)

print(f"Loaded {len(captions_df)} captions.")

# ==========================================================
# Tokenizer
# ==========================================================

tokenizer = Tokenizer()
tokenizer.build_vocab(captions_df["caption"])

print(f"Vocabulary size: {len(tokenizer)}")

# ==========================================================
# Image Transform
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ==========================================================
# Train / Val / Test Split
# ==========================================================

image_names = sorted(captions_df["image"].unique())

random.seed(CONFIG["seed"])
random.shuffle(image_names)

train_images = set(image_names[:6000])
val_images   = set(image_names[6000:7000])
test_images  = set(image_names[7000:8000])

train_df = captions_df[captions_df["image"].isin(train_images)].reset_index(drop=True)
val_df   = captions_df[captions_df["image"].isin(val_images)].reset_index(drop=True)
test_df  = captions_df[captions_df["image"].isin(test_images)].reset_index(drop=True)

print(f"Train captions : {len(train_df)}")
print(f"Val captions   : {len(val_df)}")
print(f"Test captions  : {len(test_df)}")

# ==========================================================
# Dataset
# ==========================================================

image_dir = "data/Flickr8k/Images"

train_dataset = Flickr8kDataset(
    image_dir=image_dir,
    captions_df=train_df,
    tokenizer=tokenizer,
    transform=transform,
)

val_dataset = Flickr8kDataset(
    image_dir=image_dir,
    captions_df=val_df,
    tokenizer=tokenizer,
    transform=transform,
)

test_dataset = Flickr8kDataset(
    image_dir=image_dir,
    captions_df=test_df,
    tokenizer=tokenizer,
    transform=transform,
)

# ==========================================================
# DataLoader
# ==========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
)

print("Data pipeline created successfully!")

batch = next(iter(train_loader))
images, captions, masks = batch

print(images.shape)
print(captions.shape)
print(masks.shape)

# ==========================================================
# Model
# ==========================================================

vit_encoder = ViTEncoder(
    img_size=CONFIG["image_size"],
    patch_size=CONFIG["patch_size"],
    n_embd=CONFIG["embed_dim"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["vit_depth"],
    dropout=CONFIG["dropout"],
)

text_encoder = TextEncoder(
    vocab_size=len(tokenizer),
    max_len=CONFIG["max_text_len"],
    n_embd=CONFIG["embed_dim"],
    n_head=CONFIG["n_head"],
    n_layer=CONFIG["text_depth"],
)

model = CLIPStyleModel(
    vit_encoder=vit_encoder,
    text_encoder=text_encoder,
    embed_dim=CONFIG["embed_dim"],
    projection_dim=CONFIG["projection_dim"],
    init_temperature=0.07,
).to(device)

print("Model created successfully!")

# ==========================================================
# Optimizer
# ==========================================================

optimizer = AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

print("Optimizer created.")

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=CONFIG["warmup_steps"],
)

cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=CONFIG["total_steps"] - CONFIG["warmup_steps"],
)

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        warmup_scheduler,
        cosine_scheduler,
    ],
    milestones=[
        CONFIG["warmup_steps"],
    ],
)

print("Scheduler created.")

# ==========================================================
# Forward Pass Verification
# ==========================================================

images, captions, masks = next(iter(train_loader))

images = images.to(device)
captions = captions.to(device)
masks = masks.to(device)

loss, image_features, text_features = model(
    images,
    captions,
    masks,
)

print("\nForward pass successful!")
print(f"Loss: {loss.item():.4f}")
print("Image features:", image_features.shape)
print("Text features :", text_features.shape)
