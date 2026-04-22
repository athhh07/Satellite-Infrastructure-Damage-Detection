import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class SiameseUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=5, features=[32, 64, 128, 256]):
        super().__init__()
        f = features
        self.enc1 = DoubleConv(in_ch, f[0])
        self.enc2 = DoubleConv(f[0],  f[1])
        self.enc3 = DoubleConv(f[1],  f[2])
        self.enc4 = DoubleConv(f[2],  f[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f[3], f[3]*2)
        self.up4  = nn.ConvTranspose2d(f[3]*2, f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3]*2, f[3])
        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2]*2, f[2])
        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1]*2, f[1])
        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0]*2, f[0])
        self.out_conv = nn.Conv2d(f[0], num_classes, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def forward(self, pre, post):
        pre_e1,  pre_e2,  pre_e3,  pre_e4,  pre_b  = self.encode(pre)
        post_e1, post_e2, post_e3, post_e4, post_b = self.encode(post)
        b  = torch.abs(post_b  - pre_b)
        d4 = self.dec4(torch.cat([self.up4(b),  torch.abs(post_e4-pre_e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), torch.abs(post_e3-pre_e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), torch.abs(post_e2-pre_e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), torch.abs(post_e1-pre_e1)], 1))
        return self.out_conv(d1)


class XBDDataset(Dataset):
    def __init__(self, pairs, img_size=256, augment=False):
        self.pairs   = pairs
        self.size    = img_size
        self.to_t    = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p    = self.pairs[idx]
        pre  = Image.open(p["pre"]).convert("RGB").resize((self.size, self.size))
        post = Image.open(p["post"]).convert("RGB").resize((self.size, self.size))
        mask = Image.open(p["mask"]).resize((self.size, self.size), Image.NEAREST)
        mask_t = torch.from_numpy(np.array(mask)).long().clamp(0, 4)
        return self.to_t(pre), self.to_t(post), mask_t

print("Model and Dataset classes ready.")

CHECKPOINT = "checkpoints/best_model.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

assert os.path.exists(CHECKPOINT), \
    f"Checkpoint not found at {CHECKPOINT}. Run 02_train.ipynb first."

ckpt  = torch.load(CHECKPOINT, map_location=DEVICE)
model = SiameseUNet(num_classes=5).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

IMG_SIZE = ckpt["config"]["img_size"]

print(f"Loaded checkpoint from epoch : {ckpt['epoch']}")
print(f"Best IoU during training     : {ckpt['iou']:.4f}")
print(f"Image size                   : {IMG_SIZE}")

CLASS_COLORS = {
    0: (50,  50,  50 ),   # background — dark gray
    1: (0,   200, 0  ),   # no damage  — green
    2: (255, 230, 0  ),   # minor      — yellow
    3: (255, 140, 0  ),   # major      — orange
    4: (220, 30,  30 ),   # destroyed  — red
}
CLASS_NAMES = ["Background", "No damage", "Minor damage", "Major damage", "Destroyed"]

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

LEGEND = [
    Patch(facecolor=tuple(c/255 for c in CLASS_COLORS[i]), label=CLASS_NAMES[i])
    for i in range(5)
]

print("Constants ready.")

def predict_pair(pre_path, post_path, alpha=0.45):
   
    pre_img  = Image.open(pre_path).convert("RGB")
    post_img = Image.open(post_path).convert("RGB")

    pre_t  = TRANSFORM(pre_img.resize((IMG_SIZE,IMG_SIZE))).unsqueeze(0).to(DEVICE)
    post_t = TRANSFORM(post_img.resize((IMG_SIZE,IMG_SIZE))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(pre_t, post_t)
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy()

    color_mask = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        color_mask[pred == cls] = color

    post_arr = np.array(post_img.resize((IMG_SIZE, IMG_SIZE)))
    overlay  = (post_arr * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    pre_out  = pre_img.resize((IMG_SIZE, IMG_SIZE))
    post_out = post_img.resize((IMG_SIZE, IMG_SIZE))

    return pred, color_mask, overlay, pre_out, post_out

print("predict_pair() ready.")

with open("pairs.json") as f:
    all_pairs = json.load(f)

random.seed(42)
random.shuffle(all_pairs)
split     = int(len(all_pairs) * 0.85)
val_pairs = all_pairs[split:]

# Pick 4 random val samples
sample_idx = random.sample(range(len(val_pairs)), min(4, len(val_pairs)))
n_samples  = len(sample_idx)

fig = plt.figure(figsize=(20, 5 * n_samples))

for row, si in enumerate(sample_idx):
    p = val_pairs[si]
    pred, color_mask, overlay, pre_img, post_img = predict_pair(p["pre"], p["post"])
    gt_arr = np.array(Image.open(p["mask"]).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST))

    gt_color = np.zeros((*gt_arr.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        gt_color[gt_arr == cls] = color

    titles = ["Pre-disaster", "Post-disaster", "Ground truth", "Prediction overlay"]
    imgs   = [pre_img, post_img, gt_color, overlay]

    for col, (img, title) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(n_samples, 4, row*4 + col + 1)
        ax.imshow(img)
        ax.set_title(f"{title}" if row == 0 else "")
        ax.axis("off")

fig.legend(handles=LEGEND, loc="lower center", ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()
plt.savefig("results/sample_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results/sample_predictions.png")

val_ds     = XBDDataset(val_pairs, img_size=IMG_SIZE)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

per_class_inter = np.zeros(5, dtype=np.float64)
per_class_union = np.zeros(5, dtype=np.float64)

with torch.no_grad():
    for pre, post, mask in tqdm(val_loader, desc="Computing IoU"):
        pre, post, mask = pre.to(DEVICE), post.to(DEVICE), mask.to(DEVICE)
        preds = model(pre, post).argmax(dim=1)

        for cls in range(5):
            pm = (preds == cls)
            tm = (mask  == cls)
            per_class_inter[cls] += (pm & tm).sum().item()
            per_class_union[cls] += (pm | tm).sum().item()

class_iou = per_class_inter / (per_class_union + 1e-7)
mean_iou  = class_iou[1:].mean()   # exclude background

print("\n" + "="*50)
print("  Evaluation Results")
print("="*50)
for i, name in enumerate(CLASS_NAMES):
    bar = "█" * int(class_iou[i] * 25)
    print(f"  {i}  {name:14s}: {class_iou[i]:.4f}  {bar}")
print("-"*50)
print(f"  Mean IoU (excl. background) : {mean_iou:.4f}")
print("="*50)

fig, ax = plt.subplots(figsize=(9, 4))

bar_colors = [tuple(c/255 for c in CLASS_COLORS[i]) for i in range(5)]
bars = ax.bar(CLASS_NAMES, class_iou, color=bar_colors,
              edgecolor="gray", linewidth=0.7)

ax.axhline(mean_iou, ls="--", color="navy",  lw=1.5,
           label=f"Mean IoU (no bg): {mean_iou:.4f}")
ax.axhline(0.40,     ls=":",  color="red",   lw=1.2, alpha=0.8, label="Target 40%")
ax.axhline(0.50,     ls=":",  color="green", lw=1.2, alpha=0.8, label="Target 50%")

for bar, val in zip(bars, class_iou):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_ylabel("IoU")
ax.set_title("Per-class IoU on validation set")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig("results/per_class_iou.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results/per_class_iou.png")

conf = np.zeros((5, 5), dtype=np.int64)

with torch.no_grad():
    for pre, post, mask in tqdm(val_loader, desc="Confusion matrix"):
        pre, post, mask = pre.to(DEVICE), post.to(DEVICE), mask.to(DEVICE)
        preds = model(pre, post).argmax(dim=1).cpu().numpy().flatten()
        gt    = mask.cpu().numpy().flatten()
        for t, p in zip(gt, preds):
            conf[t, p] += 1

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(conf, cmap="Blues")
plt.colorbar(im, ax=ax)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion matrix (pixel counts)")

for i in range(5):
    for j in range(5):
        v   = conf[i, j]
        txt = f"{v/1e6:.1f}M" if v >= 1_000_000 else f"{v:,}"
        col = "white" if v > conf.max()/2 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)

plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved results/confusion_matrix.png")
print("\nEvaluation complete!")
