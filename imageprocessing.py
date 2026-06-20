import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.cluster import MiniBatchKMeans
import io
import cv2

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PixelLab · Image Processing Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e6f0;
    font-family: 'Syne', sans-serif;
}

/* Hide default Streamlit chrome (menu + footer only — these are
   purely decorative and safe to fully hide) */
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Restyle the header instead of hiding it. Hiding it with
   visibility:hidden also hides the sidebar collapse/expand arrow
   that lives inside it (and Streamlit has renamed that inner
   element across versions — collapsedControl, stSidebarCollapseButton,
   etc. — so patching it back by testid is fragile). Keeping the
   header itself visible but blending it into the dark background
   works reliably across all Streamlit versions. */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.2rem !important;
}
header[data-testid="stHeader"] * {
    color: #9998b0 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]) { font-family: 'Space Mono', monospace !important; }

/* Sidebar header accent */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #7c6af7 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

/* ── Select boxes & inputs ── */
[data-baseweb="select"] > div {
    background: #14141f !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 8px !important;
    color: #e8e6f0 !important;
    font-family: 'Space Mono', monospace !important;
}
[data-baseweb="select"] svg { fill: #7c6af7 !important; }

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #7c6af7 !important;
    border-color: #7c6af7 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] {
    background: #2a2a3d !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="TrackFill"] {
    background: linear-gradient(90deg, #7c6af7, #c46af7) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a2a3d !important;
    border-radius: 12px !important;
    background: #14141f !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #7c6af7 !important;
}

[data-testid="stFileUploader"] * {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}
[data-testid="stFileUploader"] label {
    color: #9998b0 !important;
    font-size: 0.8rem !important;
}
[data-testid="stFileUploader"] small {
    font-size: 0.72rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    gap: 0.6rem !important;
    padding: 1rem !important;
}

/* Fix: previous attempts tried to repair the native button's own text
   layout, but it keeps rendering duplicate/overlapping "upload" text
   regardless — this looks like an internal Streamlit rendering glitch,
   not something fixable by adjusting font or spacing. Instead of
   fighting it, blank out whatever text the native button renders
   (whether single or duplicated) and draw one clean label on top via
   a pseudo-element, so the visible result is correct no matter what
   the native button is doing underneath. */
[data-testid="stFileUploaderDropzone"] button {
    width: 100% !important;
    margin: 0 !important;
    position: relative !important;
    color: transparent !important;
    font-size: 0 !important;
    min-height: 2.6rem !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    visibility: hidden !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "Browse files";
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #e8e6f0;
}



/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c6af7, #c46af7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Main area ── */
.block-container { padding: 2rem 2.5rem !important; max-width: 1600px !important; }

/* Hero banner */
.hero-banner {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2rem;
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, #14141f 0%, #1a1225 100%);
    border: 1px solid #2a2a3d;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(124,106,247,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon { font-size: 2.8rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #e8e6f0, #7c6af7, #c46af7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.hero-sub { font-family: 'Space Mono', monospace; font-size: 0.75rem;
    color: #6f6e85; margin: 0.25rem 0 0; letter-spacing: 0.08em; }

/* Image panels */
.img-panel {
    background: #0f0f18;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 1rem;
    height: 100%;
}
.img-panel-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    color: #6f6e85;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.img-panel-label span.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #7c6af7;
    display: inline-block;
}
.img-panel-label span.dot.green { background: #5af7a6; }

/* Info chips */
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
.chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 100px;
    background: #1a1a2e;
    border: 1px solid #2a2a3d;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #9998b0;
}
.chip .chip-val { color: #c46af7; }

/* Op badge */
.op-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    background: linear-gradient(135deg, rgba(124,106,247,0.15), rgba(196,106,247,0.1));
    border: 1px solid rgba(124,106,247,0.35);
    border-radius: 100px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #c4b5fd;
    margin-bottom: 1rem;
}

/* No-image placeholder */
.placeholder {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 280px; gap: 1rem;
    color: #3a3a52;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    border: 2px dashed #1e1e2e; border-radius: 10px;
}
.placeholder-icon { font-size: 3rem; opacity: 0.3; }

/* Divider */
hr.styled { border: none; border-top: 1px solid #1e1e2e; margin: 1.5rem 0; }

/* Metric box */
.metric-box {
    background: #14141f; border: 1px solid #1e1e2e;
    border-radius: 10px; padding: 0.9rem 1.2rem;
    font-family: 'Space Mono', monospace;
}
.metric-label { font-size: 0.65rem; color: #6f6e85; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-val { font-size: 1.1rem; color: #7c6af7; font-weight: 700; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Helper utilities ────────────────────────────────────────────────────────────

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    img_rgb = img.convert("RGB")
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# ── Processing functions ────────────────────────────────────────────────────────

def apply_threshold(img: Image.Image, threshold: int = 128) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    binary = ((arr >= threshold) * 255).astype(np.uint8)
    return Image.fromarray(binary)

def apply_kmeans(img: Image.Image, k: int = 4) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    h, w, c = arr.shape
    pixels = arr.reshape(-1, c).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.uint8)
    segmented = centers[labels].reshape(h, w, c)
    return Image.fromarray(segmented)

def apply_crop(img: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    w, h = img.size
    l = int(w * left / 100)
    t = int(h * top / 100)
    r = int(w * right / 100)
    b = int(h * bottom / 100)
    if r <= l: r = l + 1
    if b <= t: b = t + 1
    return img.crop((l, t, r, b))

def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, expand=True)

def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_negative(img: Image.Image) -> Image.Image:
    arr = 255 - np.array(img.convert("RGB"))
    return Image.fromarray(arr.astype(np.uint8))

def apply_sharpen(img: Image.Image, factor: float) -> Image.Image:
    base = img.convert("RGB")
    return ImageEnhance.Sharpness(base).enhance(factor)

def apply_laplacian(img: Image.Image) -> Image.Image:
    cv = pil_to_cv2(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return Image.fromarray(lap)

def apply_zoom(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    nw, nh = int(w * factor), int(h * factor)
    return img.resize((max(1, nw), max(1, nh)), Image.LANCZOS)

def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)

def apply_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")

def apply_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((max(1, width), max(1, height)), Image.LANCZOS)

def apply_edge_detection(img: Image.Image, low: int, high: int) -> Image.Image:
    cv = pil_to_cv2(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(edges)

# ── Operation metadata ──────────────────────────────────────────────────────────

OPERATIONS = [
    "— Select an operation —",
    "Threshold Segmentation",
    "K-Means Segmentation",
    "Cropping",
    "Rotation",
    "Blurring",
    "Negative Image",
    "Sharpening",
    "Laplacian Technique",
    "Zoom In / Zoom Out",
    "Increase / Decrease Brightness",
    "Grayscale Conversion",
    "Resize",
    "Edge Detection",
]

OP_ICONS = {
    "Threshold Segmentation":       "⬛",
    "K-Means Segmentation":         "🎨",
    "Cropping":                     "✂️",
    "Rotation":                     "🔄",
    "Blurring":                     "🌫️",
    "Negative Image":               "🔲",
    "Sharpening":                   "🔪",
    "Laplacian Technique":          "🧮",
    "Zoom In / Zoom Out":           "🔍",
    "Increase / Decrease Brightness":"☀️",
    "Grayscale Conversion":         "🩶",
    "Resize":                       "📐",
    "Edge Detection":               "🖊️",
}

# ── State ───────────────────────────────────────────────────────────────────────

if "processed" not in st.session_state:
    st.session_state.processed = None
if "last_op" not in st.session_state:
    st.session_state.last_op = None

# ── Hero banner ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-icon">🔬</div>
  <div>
    <p class="hero-title">PixelLab</p>
    <p class="hero-sub">// IMAGE PROCESSING STUDIO · REAL-TIME TRANSFORMS</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Load Image")
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        label_visibility="collapsed",
    )
    st.markdown("<hr class='styled'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Operation")
    op = st.selectbox("Operation", OPERATIONS, label_visibility="collapsed")
    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    # ── Operation-specific controls ──────────────────────────────────────────────
    params = {}

    if op == "Threshold Segmentation":
        st.markdown("**Threshold value**")
        params["threshold"] = st.slider("Threshold", 0, 255, 128, label_visibility="collapsed")

    elif op == "K-Means Segmentation":
        st.markdown("**Number of clusters (K)**")
        params["k"] = st.slider("K", 2, 16, 4, label_visibility="collapsed")

    elif op == "Cropping":
        st.markdown("**Crop boundaries (%)**")
        params["left"]   = st.slider("Left %",   0, 50,  5)
        params["top"]    = st.slider("Top %",    0, 50,  5)
        params["right"]  = st.slider("Right %",  50, 100, 95)
        params["bottom"] = st.slider("Bottom %", 50, 100, 95)

    elif op == "Rotation":
        st.markdown("**Angle (degrees)**")
        params["angle"] = st.slider("Angle", -180, 180, 45, label_visibility="collapsed")

    elif op == "Blurring":
        st.markdown("**Blur radius**")
        params["radius"] = st.slider("Radius", 1, 30, 5, label_visibility="collapsed")

    elif op == "Sharpening":
        st.markdown("**Sharpness factor**")
        params["factor"] = st.slider("Factor", 0.0, 10.0, 3.0, 0.1, label_visibility="collapsed")

    elif op == "Zoom In / Zoom Out":
        st.markdown("**Zoom factor**")
        params["factor"] = st.slider("Factor", 0.1, 5.0, 1.5, 0.1, label_visibility="collapsed")

    elif op == "Increase / Decrease Brightness":
        st.markdown("**Brightness factor**")
        params["factor"] = st.slider("Factor", 0.0, 5.0, 1.5, 0.05, label_visibility="collapsed")

    elif op == "Resize":
        st.markdown("**Target dimensions (px)**")
        params["width"]  = st.number_input("Width",  min_value=1, max_value=8000, value=512)
        params["height"] = st.number_input("Height", min_value=1, max_value=8000, value=512)

    elif op == "Edge Detection":
        st.markdown("**Canny thresholds**")
        params["low"]  = st.slider("Low threshold",  0, 255,  50)
        params["high"] = st.slider("High threshold", 0, 255, 150)

    st.markdown("<hr class='styled'>", unsafe_allow_html=True)

    apply_btn = st.button("▶  Apply Transform", use_container_width=True)

# ── Main content ─────────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="placeholder">
      <div class="placeholder-icon">🖼️</div>
      <span>Upload an image via the sidebar to get started</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load original
original = Image.open(uploaded)
ow, oh = original.size
channels = len(original.getbands())

# Apply on button press
if apply_btn and op != OPERATIONS[0]:
    with st.spinner("Processing…"):
        try:
            if op == "Threshold Segmentation":
                result = apply_threshold(original, params["threshold"])
            elif op == "K-Means Segmentation":
                result = apply_kmeans(original, params["k"])
            elif op == "Cropping":
                result = apply_crop(original, params["left"], params["top"], params["right"], params["bottom"])
            elif op == "Rotation":
                result = apply_rotation(original, params["angle"])
            elif op == "Blurring":
                result = apply_blur(original, params["radius"])
            elif op == "Negative Image":
                result = apply_negative(original)
            elif op == "Sharpening":
                result = apply_sharpen(original, params["factor"])
            elif op == "Laplacian Technique":
                result = apply_laplacian(original)
            elif op == "Zoom In / Zoom Out":
                result = apply_zoom(original, params["factor"])
            elif op == "Increase / Decrease Brightness":
                result = apply_brightness(original, params["factor"])
            elif op == "Grayscale Conversion":
                result = apply_grayscale(original)
            elif op == "Resize":
                result = apply_resize(original, params["width"], params["height"])
            elif op == "Edge Detection":
                result = apply_edge_detection(original, params["low"], params["high"])

            st.session_state.processed = result
            st.session_state.last_op = op
        except Exception as e:
            st.error(f"Processing error: {e}")

# ── Image display ────────────────────────────────────────────────────────────────
col_orig, col_proc = st.columns(2, gap="large")

with col_orig:
    st.markdown("""
    <div class="img-panel-label"><span class="dot"></span> ORIGINAL IMAGE</div>
    """, unsafe_allow_html=True)
    st.image(original, use_container_width=True)
    st.markdown(f"""
    <div class="chip-row">
      <div class="chip">W <span class="chip-val">{ow}px</span></div>
      <div class="chip">H <span class="chip-val">{oh}px</span></div>
      <div class="chip">CH <span class="chip-val">{channels}</span></div>
      <div class="chip">Mode <span class="chip-val">{original.mode}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_proc:
    st.markdown("""
    <div class="img-panel-label"><span class="dot green"></span> PROCESSED IMAGE</div>
    """, unsafe_allow_html=True)

    if st.session_state.processed is not None:
        proc = st.session_state.processed
        pw, ph = proc.size
        icon = OP_ICONS.get(st.session_state.last_op, "✨")
        st.markdown(f'<div class="op-badge">{icon} {st.session_state.last_op}</div>', unsafe_allow_html=True)
        st.image(proc, use_container_width=True)
        st.markdown(f"""
        <div class="chip-row">
          <div class="chip">W <span class="chip-val">{pw}px</span></div>
          <div class="chip">H <span class="chip-val">{ph}px</span></div>
          <div class="chip">ΔW <span class="chip-val">{pw - ow:+d}</span></div>
          <div class="chip">ΔH <span class="chip-val">{ph - oh:+d}</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        dl_bytes = image_to_bytes(proc)
        st.download_button(
            label="⬇  Download Result",
            data=dl_bytes,
            file_name=f"pixellab_{st.session_state.last_op.lower().replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.markdown("""
        <div class="placeholder">
          <div class="placeholder-icon">🔬</div>
          <span>Select an operation and click Apply</span>
        </div>
        """, unsafe_allow_html=True)

# ── Footer metrics ───────────────────────────────────────────────────────────────
st.markdown("<hr class='styled'>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
file_kb = round(uploaded.size / 1024, 1)

with m1:
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">File size</div>
      <div class="metric-val">{file_kb} KB</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">Resolution</div>
      <div class="metric-val">{ow} × {oh}</div>
    </div>""", unsafe_allow_html=True)
with m3:
    total_px = f"{(ow * oh) / 1e6:.2f}M"
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">Total pixels</div>
      <div class="metric-val">{total_px}</div>
    </div>""", unsafe_allow_html=True)
with m4:
    applied = st.session_state.last_op if st.session_state.last_op else "None"
    st.markdown(f"""
    <div class="metric-box">
      <div class="metric-label">Last operation</div>
      <div class="metric-val" style="font-size:0.85rem; color:#c46af7;">{applied}</div>
    </div>""", unsafe_allow_html=True)
