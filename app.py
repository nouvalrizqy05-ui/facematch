"""
app.py
======
Aplikasi Streamlit: Face Comparison System
Tampilan mengikuti templates/index.html (FaceMatch design)
Backend: ArcFace (512-dim) + PCA (95-dim) — Hybrid Scoring

Kelompok: Nouval · Tirta · Farritz
Mata Kuliah: Aljabar Linier | UNNES 2025
"""

import os
import io
import base64
import sys
import traceback
import warnings
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["DEEPFACE_HOME"] = "/tmp/.deepface"   # Streamlit Cloud: model disimpan di /tmp
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.scorer import HybridScorer, THRESHOLD_MATCH, WEIGHT_ONNX, WEIGHT_PCA
from src.visualizer import (
    plot_hybrid_score,
    plot_pca_variance,
    plot_embedding_comparison,
    plot_face_crops,
    plot_eigenface_overlay,
)

# ------------------------------------------------------------------
# PAGE CONFIG — minimal chrome, biarkan HTML kita yang dominan
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FaceMatch — Hybrid Verification",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Sembunyikan header/footer/menu bawaan Streamlit
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model/model_wajah.pkl"

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white", transparent=False)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()

# ------------------------------------------------------------------
# LOAD MODELS (cached)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model PCA...")
def load_scorer():
    try:
        return HybridScorer(MODEL_PATH)
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner="Memuat ArcFace embedder (bisa memakan waktu untuk download bobot model saat pertama kali)...")
def load_embedder():
    from src.embedder import FaceEmbedder
    embedder = FaceEmbedder(detector_backend="opencv")
    embedder.preload_model()
    return embedder

scorer   = load_scorer()
embedder = load_embedder()

# ------------------------------------------------------------------
# NAVBAR HTML (persis index.html)
# ------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
* { box-sizing: border-box; }
body, .stApp { background: #f5f5f4 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; color: #334155; }

nav.fm-nav {
    background: white; border-bottom: 1px solid #e2e8f0;
    position: sticky; top: 0; z-index: 50;
    padding: 0 1rem; height: 56px;
    display: flex; align-items: center; justify-content: space-between;
    max-width: 100%; margin-bottom: 0;
}
.fm-logo { display: flex; align-items: center; gap: 12px; }
.fm-logo-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #10b981, #6366f1);
    display: flex; align-items: center; justify-content: center;
}
.fm-logo-title { font-size: 16px; font-weight: 800; color: #1e293b; line-height: 1; }
.fm-logo-title span { color: #10b981; }
.fm-logo-sub { font-size: 9.5px; font-weight: 600; color: #94a3b8; }
.fm-nav-right { font-size: 10px; font-weight: 700; color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase; }

.fm-wrap { max-width: 768px; margin: 0 auto; padding: 2rem 1rem 4rem; }
.fm-center { text-align: center; margin-bottom: 1.5rem; }
.fm-title { font-size: 22px; font-weight: 800; color: #1e293b; margin-bottom: 6px; }
.fm-sub { font-size: 13px; color: #64748b; }

.fm-card {
    background: white; border-radius: 16px; padding: 20px;
    box-shadow: 0 8px 24px -8px rgba(0,0,0,0.06);
    border: 1px solid #f1f5f9; margin-bottom: 20px;
}

/* Upload zones */
.fm-upload-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.fm-zone {
    border-radius: 12px; border: 1.5px dashed #e2e8f0;
    background: #f8fafc; min-height: 150px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; cursor: pointer; transition: all 0.2s;
    overflow: hidden; position: relative;
}
.fm-zone:hover { border-color: #10b981; background: #f0fdf4; }
.fm-zone-icon {
    width: 40px; height: 40px; border-radius: 50%;
    background: white; border: 1px solid #e2e8f0;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.fm-zone-label { font-size: 12px; font-weight: 700; color: #64748b; }
.fm-zone-hint { font-size: 10px; color: #94a3b8; margin-top: 2px; }

/* Banner result */
.fm-banner {
    border-radius: 16px; padding: 24px; text-align: center; color: white;
    margin-bottom: 20px;
}
.fm-banner.match { background: linear-gradient(135deg, #10b981, #0d9488); }
.fm-banner.nomatch { background: linear-gradient(135deg, #f43f5e, #db2777); }
.fm-banner-icon {
    width: 40px; height: 40px; background: rgba(255,255,255,0.2);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; margin: 0 auto 12px;
    font-size: 20px;
}
.fm-banner h2 { font-size: 20px; font-weight: 800; margin-bottom: 4px; }
.fm-banner p { font-size: 12px; opacity: 0.8; margin-bottom: 16px; }
.fm-badges { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; }
.fm-badge {
    background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(6px); padding: 6px 14px; border-radius: 8px;
    font-size: 12px; font-weight: 700;
}

/* Plot image */
.fm-plot-wrap {
    background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 12px;
    padding: 12px; display: flex; justify-content: center;
}
.fm-plot-wrap img { max-height: 180px; object-fit: contain; cursor: zoom-in; }
.fm-plot-wrap img:hover { opacity: 0.85; }

.fm-section-title {
    font-size: 11px; font-weight: 800; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 10px;
}
.fm-desc { font-size: 11px; color: #94a3b8; margin-bottom: 12px; }

/* PCA panel grid */
.fm-pca-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.fm-pca-card { background: white; border-radius: 12px; padding: 14px; border: 1px solid #f1f5f9; }
.fm-pca-img-wrap { background: #f8fafc; border: 1px solid #f1f5f9; border-radius: 8px; height: 110px; display: flex; align-items: center; justify-content: center; margin-bottom: 8px; }
.fm-pca-img-wrap img { max-height: 100%; object-fit: contain; cursor: zoom-in; }
.fm-pca-title { font-size: 12px; font-weight: 800; color: #1e293b; margin-bottom: 2px; }
.fm-pca-mono { font-size: 10px; font-family: monospace; color: #94a3b8; background: #f8fafc; border: 1px solid #f1f5f9; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 4px; }
.fm-pca-desc { font-size: 11px; color: #64748b; }

/* Toggle */
.fm-toggle-wrap { text-align: center; margin: 4px 0 16px; }
.fm-toggle-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 18px; background: #eef2ff; color: #4f46e5;
    border: none; border-radius: 8px; font-size: 12px; font-weight: 700;
    cursor: pointer; transition: background 0.2s;
}
.fm-toggle-btn:hover { background: #e0e7ff; }

/* Footer */
.fm-footer {
    background: white; border-top: 1px solid #e2e8f0;
    padding: 16px 1rem; margin-top: 32px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 12px; color: #94a3b8;
}
.fm-footer-priv {
    display: flex; align-items: center; gap: 6px;
    color: #059669; background: #f0fdf4;
    border: 1px solid #d1fae5; padding: 6px 12px; border-radius: 99px;
    font-size: 11px; font-weight: 600;
}

/* Lightbox */
#fm-lightbox {
    display: none; position: fixed; inset: 0; z-index: 9999;
    background: rgba(15,23,42,0.88); align-items: center;
    justify-content: center; padding: 16px; cursor: zoom-out;
}
#fm-lightbox.open { display: flex; }
#fm-lightbox img { max-width: 95%; max-height: 90vh; border-radius: 12px; object-fit: contain; }
#fm-lightbox-close {
    position: absolute; top: 20px; right: 20px;
    background: rgba(255,255,255,0.15); border: none; color: white;
    width: 40px; height: 40px; border-radius: 50%; font-size: 20px;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
}
</style>

<nav class="fm-nav">
  <div class="fm-logo">
    <div class="fm-logo-icon">
      <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="white" stroke-width="2.5">
        <path stroke-linecap="round" stroke-linejoin="round" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
      </svg>
    </div>
    <div>
      <div class="fm-logo-title">FaceMatch<span>.</span></div>
      <div class="fm-logo-sub">Sistem Perbandingan Wajah masa kecil dan saat ini</div>
    </div>
  </div>
  <div class="fm-nav-right">Hybrid Verification</div>
</nav>

<div id="fm-lightbox" onclick="this.classList.remove('open')">
  <button id="fm-lightbox-close" onclick="document.getElementById('fm-lightbox').classList.remove('open')">✕</button>
  <img id="fm-lightbox-img" src="" onclick="event.stopPropagation()">
</div>
<script>
function fmOpenImg(src) {
  document.getElementById('fm-lightbox-img').src = src;
  document.getElementById('fm-lightbox').classList.add('open');
}
</script>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# WRAP OPEN
# ------------------------------------------------------------------
st.markdown('<div class="fm-wrap">', unsafe_allow_html=True)
st.markdown("""
<div class="fm-center">
  <h1 class="fm-title">Perbandingan Wajah</h1>
  <p class="fm-sub">Unggah foto masa kecil &amp; foto sekarang untuk dibandingkan secara otomatis</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# MODEL CHECK
# ------------------------------------------------------------------
if scorer is None:
    st.error("⚠️ **Model tidak ditemukan.** Pastikan `model/model_wajah.pkl` ada di folder proyek.")
    st.stop()
if not embedder.is_available():
    st.error("⚠️ **DeepFace belum terinstall.** Jalankan: `pip install deepface tf-keras`")
    st.stop()

# ------------------------------------------------------------------
# UPLOAD CARD — pakai st.file_uploader native tapi dibungkus styling
# ------------------------------------------------------------------
st.markdown('<div class="fm-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="background:#f0fdf4;border:1.5px dashed #86efac;border-radius:10px;
    padding:10px 10px 4px;text-align:center;margin-bottom:6px;">
      <div style="font-size:11px;font-weight:800;color:#15803d;margin-bottom:4px;">📸 FOTO MASA KECIL</div>
    </div>""", unsafe_allow_html=True)
    file1 = st.file_uploader("Foto masa kecil", type=["jpg","jpeg","png","bmp","webp"],
                              label_visibility="collapsed", key="f1")
    if file1:
        st.image(file1, width=300)

with col2:
    st.markdown("""
    <div style="background:#eef2ff;border:1.5px dashed #a5b4fc;border-radius:10px;
    padding:10px 10px 4px;text-align:center;margin-bottom:6px;">
      <div style="font-size:11px;font-weight:800;color:#4338ca;margin-bottom:4px;">📷 FOTO SEKARANG</div>
    </div>""", unsafe_allow_html=True)
    file2 = st.file_uploader("Foto sekarang", type=["jpg","jpeg","png","bmp","webp"],
                              label_visibility="collapsed", key="f2")
    if file2:
        st.image(file2, width=300)

st.markdown('</div>', unsafe_allow_html=True)  # tutup fm-card

# ------------------------------------------------------------------
# TOMBOL ANALISIS
# ------------------------------------------------------------------
run = False
if file1 and file2:
    run = st.button("🔍  Mulai Analisis", type="primary",
                    key="btn_run")
else:
    st.button("🔍  Mulai Analisis", type="primary",
              disabled=True, key="btn_run_dis")
    st.markdown('<p style="text-align:center;font-size:11px;color:#94a3b8;margin-top:6px;">Upload kedua foto untuk memulai</p>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------
# PROSES & HASIL
# ------------------------------------------------------------------
if run and file1 and file2:

    emb1, emb2, crop1, crop2 = None, None, None, None
    with st.spinner("Mengekstrak ArcFace embedding..."):
        try:
            img1_bgr = pil_to_bgr(Image.open(file1))
            img2_bgr = pil_to_bgr(Image.open(file2))
            emb1, crop1 = embedder.get_embedding(img1_bgr, enforce_detection=False)
            emb2, crop2 = embedder.get_embedding(img2_bgr, enforce_detection=False)
        except Exception as e:
            tb = traceback.format_exc()
            st.error(f"❌ Error saat ekstraksi embedding: {e}")
            st.code(tb, language="python")
            st.stop()

    if emb1 is None or emb2 is None:
        st.markdown("""
        <div class="fm-card" style="border-left:4px solid #f59e0b;">
          <p style="color:#b45309;font-weight:700;margin:0;">⚠️ Wajah tidak terdeteksi</p>
          <p style="font-size:12px;color:#64748b;margin:6px 0 0;">
            Pastikan wajah terlihat jelas, pencahayaan cukup, dan menghadap ke depan.
            Coba foto dengan resolusi lebih tinggi.
          </p>
        </div>""", unsafe_allow_html=True)
        st.stop()

    try:
        result = scorer.score(emb1, emb2)
    except Exception as e:
        tb = traceback.format_exc()
        st.error(f"❌ Gagal menghitung skor: {e}")
        st.code(tb, language="python")
        st.stop()

    # --- Generate semua plot ---
    with st.spinner("Membuat visualisasi..."):
        try:
            b64_crops  = fig_to_b64(plot_face_crops(crop1, crop2, result, "Foto Masa Kecil", "Foto Sekarang"))
            b64_eigen  = fig_to_b64(plot_eigenface_overlay(crop1, crop2, scorer._pca, emb1, emb2, result))
            b64_hybrid = fig_to_b64(plot_hybrid_score(result))
            b64_pca2d  = fig_to_b64(plot_embedding_comparison(emb1, emb2, scorer._pca, "Foto Masa Kecil", "Foto Sekarang"))
            b64_pca_var= fig_to_b64(plot_pca_variance(scorer._pca))
        except Exception as e:
            tb = traceback.format_exc()
            st.error(f"❌ Gagal membuat visualisasi: {e}")
            st.code(tb, language="python")
            st.stop()

    match_class = "match" if result.is_match else "nomatch"
    match_icon  = "✅" if result.is_match else "❌"
    match_title = "Identitas Cocok" if result.is_match else "Tidak Identik"
    match_desc  = "Kedua foto dikonfirmasi sebagai individu yang sama." if result.is_match \
                  else "Kedua foto diprediksi merupakan individu yang berbeda."

    # --- BANNER ---
    st.markdown(f"""
    <div class="fm-banner {match_class}">
      <div class="fm-banner-icon">{match_icon}</div>
      <h2>{match_title}</h2>
      <p>{match_desc}</p>
      <div class="fm-badges">
        <span class="fm-badge">Skor: {result.hybrid_score*100:.1f}%</span>
        <span class="fm-badge">Dist: {result.eucl_dist:.3f}</span>
        <span class="fm-badge">{result.confidence_icon} {result.confidence}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- DETEKSI & ALIGNMENT ---
    st.markdown("""
    <div class="fm-card">
      <div class="fm-section-title">Deteksi &amp; Penyelarasan Wajah</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="fm-plot-wrap">
        <img src="data:image/png;base64,{b64_crops}" onclick="fmOpenImg(this.src)">
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- EIGENFACE ANALYSIS ---
    st.markdown("""
    <div class="fm-card">
      <div class="fm-section-title">Eigenface Analysis</div>
      <p class="fm-desc">Visualisasi grayscale overlay dan peta perbedaan piksel untuk mencocokkan struktur geometri wajah.</p>
    """, unsafe_allow_html=True)
    st.markdown(f"""
      <div class="fm-plot-wrap">
        <img src="data:image/png;base64,{b64_eigen}" onclick="fmOpenImg(this.src)">
      </div>
    </div>
    """, unsafe_allow_html=True)

    # --- TOGGLE PCA ---
    st.markdown("""
    <div class="fm-toggle-wrap">
      <button class="fm-toggle-btn" onclick="
        var p=document.getElementById('fm-pca-panel');
        p.style.display = p.style.display==='none' ? 'block' : 'none';
        this.querySelector('.fm-chev').style.transform =
          p.style.display==='none' ? 'rotate(0deg)' : 'rotate(180deg)';
      ">
        <svg class="fm-chev" width="16" height="16" fill="none" viewBox="0 0 24 24"
          stroke="currentColor" stroke-width="2.5"
          style="transition:transform 0.3s;">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/>
        </svg>
        Lihat Detail Visualisasi PCA
      </button>
    </div>

    <div id="fm-pca-panel" style="display:none;">
      <div class="fm-pca-grid">
        <div class="fm-pca-card">
          <div class="fm-pca-img-wrap">
            <img src="data:image/png;base64,{pca2d}" onclick="fmOpenImg(this.src)">
          </div>
          <div class="fm-pca-title">Proyeksi PCA 2D</div>
          <div class="fm-pca-mono">(PC1, PC2)</div>
          <div class="fm-pca-desc">Kedekatan posisi titik = kemiripan identitas.</div>
        </div>
        <div class="fm-pca-card">
          <div class="fm-pca-img-wrap">
            <img src="data:image/png;base64,{hybrid}" onclick="fmOpenImg(this.src)">
          </div>
          <div class="fm-pca-title">Hybrid Score</div>
          <div class="fm-pca-mono">0.7·Cos_A + 0.3·Cos_P</div>
          <div class="fm-pca-desc">Dekomposisi skor ArcFace dan PCA.</div>
        </div>
        <div class="fm-pca-card">
          <div class="fm-pca-img-wrap">
            <img src="data:image/png;base64,{var_}" onclick="fmOpenImg(this.src)">
          </div>
          <div class="fm-pca-title">Komponen PCA</div>
          <div class="fm-pca-mono">Var(k) ≥ 95%</div>
          <div class="fm-pca-desc">Jumlah dimensi optimal dari SVD.</div>
        </div>
      </div>
    </div>
    """.replace("{pca2d}", b64_pca2d).replace("{hybrid}", b64_hybrid).replace("{var_}", b64_pca_var),
    unsafe_allow_html=True)

    # --- RESET ---
    st.markdown('<div style="text-align:center;margin-top:24px;">', unsafe_allow_html=True)
    if st.button("↩  Analisis Ulang", key="btn_reset"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# WRAP CLOSE + FOOTER
# ------------------------------------------------------------------
st.markdown('</div>', unsafe_allow_html=True)  # tutup fm-wrap

st.markdown("""
<div class="fm-footer">
  <span>© 2026 FaceMatch. All rights reserved.</span>
  <div class="fm-footer-priv">
    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
      <path stroke-linecap="round" stroke-linejoin="round"
        d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
    </svg>
    Privasi Terjamin: Foto Anda aman dan tidak disimpan di server.
  </div>
</div>
""", unsafe_allow_html=True)
