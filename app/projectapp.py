import io
import os
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms as T
 
from model import SiameseUNet
from utils import (
    preprocess, to_color_mask, make_overlay,
    damage_stats, CLASS_NAMES, CLASS_COLORS, IMG_SIZE,
)
 
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Satellite Damage Detection",
    page_icon="🛰️",
    layout="wide",
)
 
# ── Title ─────────────────────────────────────────────────────
st.title("🛰️ Satellite Infrastructure Damage Detection")
st.write("Upload a **pre-disaster** and **post-disaster** satellite image pair to detect and visualize damage.")
st.divider()
 
# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
 
checkpoint_path = st.sidebar.text_input(
    "Checkpoint path",
    value="../checkpoints/best_model.pth"
)
 
alpha = st.sidebar.slider(
    "Overlay opacity",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Controls how strongly the damage colors show over the image"
)
 
show_raw = st.sidebar.checkbox(
    "Show mask only (no overlay)",
    value=False
)
 
st.sidebar.divider()
st.sidebar.subheader("🎨 Damage Color Legend")
st.sidebar.markdown("""
| Color | Class |
|-------|-------|
| ⬛ Dark gray | Background |
| 🟩 Green | No damage |
| 🟨 Yellow | Minor damage |
| 🟧 Orange | Major damage |
| 🟥 Red | Destroyed |
""")
 
st.sidebar.divider()
st.sidebar.subheader("ℹ️ About")
st.sidebar.info(
    "**Model:** Siamese U-Net\n\n"
    "**Dataset:** xBD (2,799 image pairs)\n\n"
    "**Framework:** PyTorch + Streamlit\n\n"
    "**Project:** Final Year BE Data Science"
)
 
# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    """Load trained model from checkpoint file."""
    if not os.path.exists(path):
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(path, map_location=device)
    model  = SiameseUNet(num_classes=5)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, device, ckpt
 
model, device, ckpt = load_model(checkpoint_path)
 
# Show model status in sidebar
if model is not None:
    st.sidebar.success(
        f"✅ Model loaded successfully!\n\n"
        f"Trained for **{ckpt['epoch']}** epochs\n\n"
        f"Best IoU: **{ckpt['iou']:.4f}**"
    )
else:
    st.sidebar.error(
        "❌ Model not found!\n\n"
        "Please check the checkpoint path above."
    )
 
# ── Image upload ──────────────────────────────────────────────
st.subheader("📤 Upload Images")
 
col1, col2 = st.columns(2)
 
with col1:
    st.markdown("**Pre-disaster Image**")
    st.caption("Satellite image BEFORE the disaster")
    pre_file = st.file_uploader(
        "Upload pre-disaster image",
        type=["png", "jpg", "jpeg"],
        key="pre",
        label_visibility="collapsed"
    )
    if pre_file is not None:
        st.image(pre_file, caption="Pre-disaster", use_column_width=True)
 
with col2:
    st.markdown("**Post-disaster Image**")
    st.caption("Satellite image AFTER the disaster")
    post_file = st.file_uploader(
        "Upload post-disaster image",
        type=["png", "jpg", "jpeg"],
        key="post",
        label_visibility="collapsed"
    )
    if post_file is not None:
        st.image(post_file, caption="Post-disaster", use_column_width=True)
 
st.divider()
 
# ── Run button ────────────────────────────────────────────────
run_clicked = st.button(
    "🚀 Run Damage Detection",
    type="primary",
    use_container_width=True
)
 
# ── Inference ─────────────────────────────────────────────────
if run_clicked:
 
    # Check inputs
    if pre_file is None or post_file is None:
        st.warning("⚠️ Please upload both images before running detection.")
 
    elif model is None:
        st.error(
            "❌ Model checkpoint not found. "
            "Make sure `best_model.pth` is inside the `checkpoints/` folder."
        )
 
    else:
        # Load images
        pre_img  = Image.open(pre_file).convert("RGB")
        post_img = Image.open(post_file).convert("RGB")
 
        # Run model
        with st.spinner("🔍 Running damage detection... please wait"):
 
            pre_tensor  = preprocess(pre_img).to(device)
            post_tensor = preprocess(post_img).to(device)
 
            with torch.no_grad():
                output = model(pre_tensor, post_tensor)
                pred   = output.argmax(dim=1).squeeze().cpu().numpy()
 
        st.success("✅ Detection complete!")
 
        # Build output images
        color_mask = to_color_mask(pred)
        overlay    = make_overlay(post_img, color_mask, alpha=alpha)
        stats      = damage_stats(pred)
 
        # ── Results ───────────────────────────────────────────
        st.subheader("📊 Results")
 
        r1, r2, r3 = st.columns(3)
 
        with r1:
            st.image(
                pre_img.resize((IMG_SIZE, IMG_SIZE)),
                caption="Pre-disaster",
                use_column_width=True
            )
 
        with r2:
            st.image(
                post_img.resize((IMG_SIZE, IMG_SIZE)),
                caption="Post-disaster",
                use_column_width=True
            )
 
        with r3:
            if show_raw:
                st.image(color_mask, caption="Damage mask", use_column_width=True)
            else:
                st.image(overlay, caption="Damage overlay", use_column_width=True)
 
        st.divider()
 
        # ── Damage statistics ─────────────────────────────────
        st.subheader("📈 Damage Statistics")
        st.caption("Percentage of pixels belonging to each damage class")
 
        c1, c2, c3, c4, c5 = st.columns(5)
 
        for col, (name, data) in zip([c1,c2,c3,c4,c5], stats.items()):
            col.metric(
                label=name,
                value=f"{data['pct']:.1f}%",
                help=f"{data['count']:,} pixels"
            )
 
        st.divider()
 
        # ── Simple bar chart ──────────────────────────────────
        st.subheader("📉 Damage Distribution Chart")
 
        import matplotlib.pyplot as plt
 
        labels = list(stats.keys())
        values = [stats[k]["pct"] for k in labels]
        colors = ["#323232", "#00c800", "#ffe600", "#ff8c00", "#dc1e1e"]
 
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
 
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9
            )
 
        ax.set_xlabel("Percentage of pixels (%)")
        ax.set_title("Damage class distribution")
        ax.set_xlim(0, max(values) + 5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
        st.divider()
 
        # ── Download section ──────────────────────────────────
        st.subheader("⬇️ Download Results")
 
        def pil_to_bytes(arr):
            buf = io.BytesIO()
            Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
            return buf.getvalue()
 
        csv_data = "class,pixels,percentage\n" + "\n".join(
            f"{name},{data['count']},{data['pct']}"
            for name, data in stats.items()
        )
 
        d1, d2, d3 = st.columns(3)
 
        with d1:
            st.download_button(
                label="📥 Download Overlay Image",
                data=pil_to_bytes(overlay),
                file_name="damage_overlay.png",
                mime="image/png",
                use_container_width=True
            )
 
        with d2:
            st.download_button(
                label="📥 Download Mask Image",
                data=pil_to_bytes(color_mask),
                file_name="damage_mask.png",
                mime="image/png",
                use_container_width=True
            )
 
        with d3:
            st.download_button(
                label="📥 Download Stats (CSV)",
                data=csv_data,
                file_name="damage_stats.csv",
                mime="text/csv",
                use_container_width=True
            )