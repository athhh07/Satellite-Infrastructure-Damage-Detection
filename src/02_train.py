"""
## Model Training 
"""

"""
### imports
"""

import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import warnings
warnings.filterwarnings("ignore")

"""
### config
"""

CONFIG = {
    "img_size"   : 256,
    "batch_size" : 4,
    "num_epochs" : 30,
    "lr"         : 1e-4,
    "num_classes": 5,
    "pairs_json" : "pairs.json",
    "val_split"  : 0.15,
    "checkpoint" : "checkpoints/best_model.pth",
    "seed"       : 42,

    "max_train_samples": 400, 
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

"""
### Dataset class
"""

class XBDDataset(Dataset):
    def __init__(self, pairs, img_size=256, augment=False):
        self.pairs   = pairs
        self.size    = img_size
        self.augment = augment

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        pre  = Image.open(p["pre"]).convert("RGB").resize((self.size, self.size))
        post = Image.open(p["post"]).convert("RGB").resize((self.size, self.size))
        mask = Image.open(p["mask"]).resize((self.size, self.size), Image.NEAREST)

        # Augmentation — same transform applied to all three
        if self.augment:
            if random.random() > 0.5:
                pre  = TF.hflip(pre)
                post = TF.hflip(post)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                pre  = TF.vflip(pre)
                post = TF.vflip(post)
                mask = TF.vflip(mask)
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                pre  = TF.rotate(pre,  k * 90)
                post = TF.rotate(post, k * 90)
                mask = TF.rotate(mask, k * 90)

        pre_t  = self.to_tensor(pre)
        post_t = self.to_tensor(post)
        mask_t = torch.from_numpy(np.array(mask)).long().clamp(0, 4)

        return pre_t, post_t, mask_t

print("XBDDataset class defined.")

"""
### train/ val split
"""

with open(CONFIG["pairs_json"]) as f:
    all_pairs = json.load(f)

random.shuffle(all_pairs)

"""
### Model -Siamese UNet
"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=5, features=[32, 64, 128, 256]):
        super(SiameseUNet, self).__init__()
        f = features

        # Encoder 
        self.enc1 = DoubleConv(in_ch, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(f[3], f[3] * 2)

        # Decoder
        self.up4  = nn.ConvTranspose2d(f[3]*2, f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3]*2, f[3])

        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2]*2, f[2])

        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1]*2, f[1])

        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0]*2, f[0])

        self.out_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def forward(self, pre, post):
        pre_e1, pre_e2, pre_e3, pre_e4, pre_b = self.encode(pre)
        post_e1, post_e2, post_e3, post_e4, post_b = self.encode(post)

        # Difference features
        b  = torch.abs(post_b - pre_b)
        d4 = self.dec4(torch.cat([self.up4(b), torch.abs(post_e4 - pre_e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), torch.abs(post_e3 - pre_e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), torch.abs(post_e2 - pre_e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), torch.abs(post_e1 - pre_e1)], dim=1))

        return self.out_conv(d1)


# Sanity check
_m   = SiameseUNet(num_classes=5)
_pre = torch.randn(1, 3, 256, 256)
_out = _m(_pre, _pre)

assert _out.shape == (1, 5, 256, 256), f"Unexpected shape: {_out.shape}"

params = sum(p.numel() for p in _m.parameters() if p.requires_grad)

print(f"Model output shape   : {_out.shape}  ✓")
print(f"Trainable parameters : {params:,}")

"""
### IoU Metric
"""

def compute_iou(preds, targets, num_classes=5, ignore_background=True):
    if preds.dim() == 4:
        preds = preds.argmax(dim=1)

    start = 1 if ignore_background else 0
    ious  = []
    for cls in range(start, num_classes):
        pred_m   = (preds   == cls)
        target_m = (targets == cls)
        inter = (pred_m & target_m).sum().float()
        union = (pred_m | target_m).sum().float()
        if union > 0:
            ious.append((inter / union).item())

    return float(np.mean(ious)) if ious else 0.0

print("Success!!")

"""
### Loss function
"""

CLASS_WEIGHTS = torch.tensor(
    [0.2,   
     1.0,   
     2.0,   
     3.0,   
     4.0],  
    dtype=torch.float
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
print("Weighted CrossEntropyLoss ready.")
print(f"Weights: {CLASS_WEIGHTS.tolist()}")


"""
### training loop
"""

model     = SiameseUNet(num_classes=CONFIG["num_classes"]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5
)

best_val_iou = 0.0
train_losses = []
val_ious     = []

print(f"Starting training — {CONFIG['num_epochs']} epochs\n")

for epoch in range(1, CONFIG["num_epochs"] + 1):

    # Train 
    model.train()
    epoch_loss = 0.0

    for pre, post, mask in tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d}/{CONFIG['num_epochs']} [train]",
            leave=False):

        pre, post, mask = pre.to(DEVICE), post.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        logits = model(pre, post)
        loss   = criterion(logits, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validate
    model.eval()
    batch_ious = []

    with torch.no_grad():
        for pre, post, mask in val_loader:
            pre, post, mask = pre.to(DEVICE), post.to(DEVICE), mask.to(DEVICE)
            logits = model(pre, post)
            batch_ious.append(compute_iou(logits, mask, CONFIG["num_classes"]))

    avg_iou = float(np.mean(batch_ious))
    val_ious.append(avg_iou)
    scheduler.step(avg_iou)

    # Save best model
    saved = ""
    if avg_iou > best_val_iou:
        best_val_iou = avg_iou
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "iou"        : best_val_iou,
            "config"     : CONFIG,
        }, CONFIG["checkpoint"])
        saved = "  ✓ saved"

    print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
          f"Val IoU: {avg_iou:.4f} | Best: {best_val_iou:.4f}{saved}")

print(f"\nTraining complete.  Best Val IoU: {best_val_iou:.4f}")

epochs = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(epochs, train_losses, color="steelblue", linewidth=2)
ax1.set_title("Training loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(alpha=0.3)

ax2.plot(epochs, val_ious, color="darkorange", linewidth=2, label="Val IoU")
ax2.axhline(0.40, ls="--", color="red",   alpha=0.6, label="Target 40%")
ax2.axhline(0.50, ls="--", color="green", alpha=0.6, label="Target 50%")
ax2.set_title("Validation IoU")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean IoU (excl. background)")
ax2.set_ylim(0, 1)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results/training_curves.png")
