# app/app.py
# Run with:  streamlit run app.py
# Place this inside:  ..\Satellite Infrastructure Damage Detection\app\

import io
import os
import torch
import numpy as np
import streamlit as st
from PIL import Image

from model import SiameseUNet
from utils import (
    preprocess, to_color_mask, make_overlay,
    damage_stats, CLASS_NAMES, CLASS_COLORS, IMG_SIZE,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SatDamage — Damage Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  #MainMenu, footer, header  { visibility: hidden; }

  .stApp { background: #07090f; }

  [data-testid="stSidebar"] {
    background: #0b0e18;
    border-right: 1px solid #151d30;
  }

  [data-testid="stFileUploader"] {
    background: #0b0e18;
    border: 1.5px dashed #1c2d48;
    border-radius: 14px;
    transition: border-color .25s;
  }
  [data-testid="stFileUploader"]:hover { border-color: #2563eb; }

  .stButton > button[kind="primary"] {
    background: #1d4ed8;
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: .75rem 2rem;
    font-family: 'Space Mono', monospace;
    font-size: .88rem;
    letter-spacing: .06em;
    font-weight: 700;
    width: 100%;
    transition: all .2s;
  }
  .stButton > button[kind="primary"]:hover {
    background: #1e40af;
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(29,78,216,.4);
  }

  .stDownloadButton > button {
    background: #0b0e18 !important;
    border: 1px solid #1c2d48 !important;
    color: #60a5fa !important;
    border-radius: 8px !important;
    font-size: .82rem !important;
    width: 100%;
    transition: all .2s;
    font-family: 'Space Mono', monospace !important;
  }
  .stDownloadButton > button:hover {
    border-color: #3b82f6 !important;
    background: #0f1929 !important;
  }

  .stTextInput > div > input {
    background: #0b0e18 !important;
    border: 1px solid #1c2d48 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .78rem !important;
  }

  hr { border-color: #151d30 !important; margin: 1.4rem 0 !important; }

  .mono { font-family: 'Space Mono', monospace; }
  .tag  {
    display: inline-block;
    background: rgba(37,99,235,.12);
    border: 1px solid rgba(37,99,235,.3);
    color: #60a5fa;
    border-radius: 20px;
    padding: .15rem .7rem;
    font-size: .72rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: .06em;
    margin-right: .3rem;
  }

  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: .68rem;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: .14em;
    margin-bottom: .6rem;
  }

  .img-label {
    font-family: 'Space Mono', monospace;
    font-size: .65rem;
    color: #334155;
    text-align: center;
    margin-top: .35rem;
    text-transform: uppercase;
    letter-spacing: .1em;
  }

  .stat-card {
    background: #0b0e18;
    border: 1px solid #151d30;
    border-radius: 12px;
    padding: .9rem .7rem;
    text-align: center;
  }
  .stat-lbl {
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: .3rem;
  }
  .stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1;
  }
  .stat-px {
    font-family: 'Space Mono', monospace;
    font-size: .6rem;
    color: #334155;
    margin-top: .25rem;
  }

  .info-panel {
    background: #080d1a;
    border: 1px solid #1c2d48;
    border-left: 3px solid #2563eb;
    border-radius: 8px;
    padding: .9rem 1.1rem;
    font-size: .86rem;
    color: #64748b;
    line-height: 1.7;
  }

  .summary-bar {
    background: #080d1a;
    border: 1px solid #1c2d48;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: .8rem;
    font-size: .86rem;
    color: #94a3b8;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.8rem 0 .8rem">
  <div style="margin-bottom:.5rem">
    <span class="tag">🛰 Remote Sensing</span>
    <span class="tag">PyTorch</span>
    <span class="tag">Siamese U-Net</span>
    <span class="tag">xBD Dataset</span>
  </div>
  <h1 style="font-family:'Space Mono',monospace;font-size:clamp(1.3rem,3vw,1.9rem);
             font-weight:700;color:#f1f5f9;margin:.4rem 0 .3rem;line-height:1.2;
             letter-spacing:-.02em;">
    Satellite Infrastructure<br>
    <span style="color:#3b82f6">Damage Detection</span>
  </h1>
  <p style="color:#334155;font-size:.86rem;margin:0">
    Upload a pre/post disaster image pair → get a pixel-level damage map in seconds
  </p>
</div>
<hr>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:.68rem;
                color:#2563eb;letter-spacing:.12em;text-transform:uppercase;
                margin:.5rem 0 1rem">⚙ Settings</div>
    """, unsafe_allow_html=True)

    ckpt_path = st.text_input("Checkpoint path",
                              value="../checkpoints/best_model.pth")
    alpha     = st.slider("Overlay opacity", .10, .90, .45, .05)
    raw_mask  = st.checkbox("Show raw mask only", False)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:.68rem;
                color:#2563eb;letter-spacing:.12em;text-transform:uppercase;
                margin-bottom:.7rem">◈ Legend</div>
    """, unsafe_allow_html=True)

    for hex_c, name in [
        ("#2d3748","Background"),
        ("#22c55e","No damage"),
        ("#eab308","Minor damage"),
        ("#f97316","Major damage"),
        ("#ef4444","Destroyed"),
    ]:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:9px;'
            f'margin:5px 0;font-size:.83rem;color:#64748b;">'
            f'<div style="width:11px;height:11px;background:{hex_c};'
            f'border-radius:2px;flex-shrink:0"></div>{name}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.75rem;color:#1e293b;line-height:1.8;
                font-family:'Space Mono',monospace;">
      Model · Siamese U-Net<br>
      Data  · xBD 2,799 pairs<br>
      Res   · 256 × 256 px<br>
      FW    · PyTorch + Streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(path, map_location=device)
    m      = SiameseUNet(num_classes=5)
    m.load_state_dict(ckpt["model_state"])
    m.eval().to(device)
    return m, device, ckpt

model, device, ckpt = load_model(ckpt_path)

with st.sidebar:
    if model:
        st.markdown(
            f'<div style="background:#031a09;border:1px solid #14532d;'
            f'border-radius:8px;padding:.7rem .9rem;margin-top:.3rem;'
            f'font-family:Space Mono,monospace;font-size:.72rem;'
            f'color:#4ade80;line-height:1.7">'
            f'✓ Loaded — Epoch {ckpt["epoch"]}<br>'
            f'<span style="color:#86efac">Best IoU: {ckpt["iou"]:.4f}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#1a0505;border:1px solid #7f1d1d;'
            'border-radius:8px;padding:.7rem .9rem;margin-top:.3rem;'
            'font-size:.72rem;color:#fca5a5;font-family:Space Mono,monospace;">'
            '✗ Checkpoint not found<br>'
            '<span style="color:#f87171">Set correct path above</span></div>',
            unsafe_allow_html=True,
        )

# ── Uploads ───────────────────────────────────────────────────
st.markdown('<div class="section-label">◈ Input Images</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown(
        '<div class="mono" style="font-size:.7rem;color:#334155;'
        'margin-bottom:.4rem;letter-spacing:.05em">01 / PRE-DISASTER</div>',
        unsafe_allow_html=True)
    pre_file = st.file_uploader("pre", type=["png","jpg","jpeg"],
                                key="pre", label_visibility="collapsed")
    if pre_file:
        st.image(pre_file, use_column_width=True)
        st.markdown('<div class="img-label">Pre-disaster ✓</div>', unsafe_allow_html=True)

with col2:
    st.markdown(
        '<div class="mono" style="font-size:.7rem;color:#334155;'
        'margin-bottom:.4rem;letter-spacing:.05em">02 / POST-DISASTER</div>',
        unsafe_allow_html=True)
    post_file = st.file_uploader("post", type=["png","jpg","jpeg"],
                                 key="post", label_visibility="collapsed")
    if post_file:
        st.image(post_file, use_column_width=True)
        st.markdown('<div class="img-label">Post-disaster ✓</div>', unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
run = st.button("⟶  Run Damage Detection", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Inference ─────────────────────────────────────────────────
if run:
    if not pre_file or not post_file:
        st.warning("Upload both images before running detection.")
    elif not model:
        st.error("Checkpoint not found. Place `best_model.pth` in `checkpoints/`.")
    else:
        pre_img  = Image.open(pre_file).convert("RGB")
        post_img = Image.open(post_file).convert("RGB")

        with st.spinner("Running inference..."):
            with torch.no_grad():
                pred = model(
                    preprocess(pre_img).to(device),
                    preprocess(post_img).to(device)
                ).argmax(dim=1).squeeze().cpu().numpy()

        color_mask = to_color_mask(pred)
        overlay    = make_overlay(post_img, color_mask, alpha=alpha)
        stats      = damage_stats(pred)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Results
        st.markdown('<div class="section-label">◈ Detection Output</div>',
                    unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3, gap="medium")
        with r1:
            st.image(pre_img.resize((IMG_SIZE,IMG_SIZE)), use_column_width=True)
            st.markdown('<div class="img-label">Pre-disaster</div>', unsafe_allow_html=True)
        with r2:
            st.image(post_img.resize((IMG_SIZE,IMG_SIZE)), use_column_width=True)
            st.markdown('<div class="img-label">Post-disaster</div>', unsafe_allow_html=True)
        with r3:
            st.image(color_mask if raw_mask else overlay, use_column_width=True)
            st.markdown(
                f'<div class="img-label">{"Damage mask" if raw_mask else "Damage overlay"}</div>',
                unsafe_allow_html=True)

        # Stats
        st.markdown('<div class="section-label" style="margin-top:1.2rem">'
                    '◈ Damage Statistics</div>', unsafe_allow_html=True)

        CHEX = {
            "Background"  : "#475569",
            "No damage"   : "#22c55e",
            "Minor damage": "#eab308",
            "Major damage": "#f97316",
            "Destroyed"   : "#ef4444",
        }

        sc = st.columns(5, gap="small")
        for col, (name, data) in zip(sc, stats.items()):
            col.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-lbl">{name}</div>'
                f'<div class="stat-val" style="color:{CHEX[name]}">'
                f'{data["pct"]:.1f}%</div>'
                f'<div class="stat-px">{data["count"]:,} px</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Summary
        damaged_pct = sum(stats[k]["pct"]
                          for k in ["Minor damage","Major damage","Destroyed"])
        worst = max(
            [(n, stats[n]["pct"]) for n in ["Minor damage","Major damage","Destroyed"]],
            key=lambda x: x[1]
        )
        st.markdown(
            f'<div class="summary-bar">'
            f'<b style="color:#e2e8f0;font-family:Space Mono,monospace;'
            f'font-size:.8rem">ANALYSIS SUMMARY</b><br>'
            f'Total damaged area: <b style="color:#f97316">{damaged_pct:.1f}%</b> '
            f'of image &nbsp;·&nbsp; '
            f'Dominant class: <b style="color:{CHEX[worst[0]]}">'
            f'{worst[0]}</b> at {worst[1]:.1f}%'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Downloads
        st.markdown('<div class="section-label" style="margin-top:1.2rem">'
                    '◈ Export Results</div>', unsafe_allow_html=True)

        def to_bytes(arr):
            buf = io.BytesIO()
            Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
            return buf.getvalue()

        csv = "class,pixels,percentage\n" + "\n".join(
            f"{n},{d['count']},{d['pct']}" for n, d in stats.items()
        )

        d1, d2, d3 = st.columns(3, gap="small")
        with d1:
            st.download_button("⬇ Overlay PNG", to_bytes(overlay),
                               "damage_overlay.png", "image/png",
                               use_container_width=True)
        with d2:
            st.download_button("⬇ Mask PNG", to_bytes(color_mask),
                               "damage_mask.png", "image/png",
                               use_container_width=True)
        with d3:
            st.download_button("⬇ Stats CSV", csv,
                               "damage_stats.csv", "text/csv",
                               use_container_width=True)

elif not pre_file and not post_file:
    st.markdown("""
    <div class="info-panel" style="margin-top:.5rem">
      <b style="color:#e2e8f0">How to use</b><br>
      1. Upload a <b style="color:#93c5fd">pre-disaster</b> satellite image on the left<br>
      2. Upload the same area <b style="color:#93c5fd">post-disaster</b> on the right<br>
      3. Click <b style="color:#93c5fd">Run Damage Detection</b><br>
      4. View the colour-coded overlay and download results
    </div>
    """, unsafe_allow_html=True)
