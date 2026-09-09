"""
Peripheral Blood Cell (PBC) Analyzer — Streamlit Application
--------------------------------------------------------------
Dark Red & Black Theme | Deep Learning VGG16 Image Classification
Fully fixed HTML rendering (using st.html)
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HEMA-AI | Blood Cell Classifier",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. BLACK & RED THEME CSS STYLING
# =========================================================

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0b0b0e;
        color: #e2e8f0;
    }

    .hero-header {
        background: linear-gradient(135deg, #1c0509 0%, #0b0b0e 60%, #2b080e 100%);
        border: 1px solid rgba(255, 46, 77, 0.25);
        border-radius: 16px;
        padding: 32px 28px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(255, 46, 77, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -20%;
        width: 140%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 46, 77, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        letter-spacing: -0.5px;
    }

    .hero-title span {
        color: #ff2e4d;
        text-shadow: 0 0 15px rgba(255, 46, 77, 0.5);
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 800px;
        line-height: 1.6;
    }

    .custom-card {
        background: rgba(22, 22, 29, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 46, 77, 0.2);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .custom-card:hover {
        border-color: rgba(255, 46, 77, 0.45);
        box-shadow: 0 6px 24px rgba(255, 46, 77, 0.15);
    }

    .result-box {
        background: linear-gradient(145deg, #180609 0%, #0d0d12 100%);
        border: 2px solid #ff2e4d;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 35px rgba(255, 46, 77, 0.25);
        margin-bottom: 24px;
    }

    .result-badge {
        display: inline-block;
        background: rgba(255, 46, 77, 0.15);
        color: #ff4d6d;
        border: 1px solid #ff2e4d;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .result-class {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0;
    }

    .result-confidence {
        font-size: 1.3rem;
        color: #ff2e4d;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .prob-bar-container {
        margin-bottom: 14px;
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .prob-name {
        color: #cbd5e1;
    }

    .prob-val {
        color: #ff4d6d;
        font-family: 'JetBrains Mono', monospace;
    }

    .prob-track {
        background: #1a1a24;
        border-radius: 8px;
        height: 10px;
        width: 100%;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .prob-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #99001a 0%, #ff2e4d 100%);
        box-shadow: 0 0 10px rgba(255, 46, 77, 0.5);
        transition: width 0.6s ease;
    }

    .prob-fill-top {
        background: linear-gradient(90deg, #cc0022 0%, #ff4d6d 100%);
    }

    .stButton > button {
        background: linear-gradient(135deg, #cc0022 0%, #ff2e4d 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 18px rgba(255, 46, 77, 0.35) !important;
        transition: all 0.25s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 26px rgba(255, 46, 77, 0.55) !important;
        background: linear-gradient(135deg, #e60026 0%, #ff4d6d 100%) !important;
    }

    section[data-testid="stFileUploadDropzone"] {
        background-color: rgba(22, 22, 29, 0.8) !important;
        border: 2px dashed rgba(255, 46, 77, 0.4) !important;
        border-radius: 14px !important;
        padding: 24px !important;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #ff2e4d !important;
        background-color: rgba(35, 15, 22, 0.8) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #07070a !important;
        border-right: 1px solid rgba(255, 46, 77, 0.15) !important;
    }

    .sidebar-section-title {
        color: #ff2e4d;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .metric-card {
        background: #14141c;
        border: 1px solid rgba(255, 46, 77, 0.2);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }

    .metric-val {
        color: #ff2e4d;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .metric-lbl {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #121218;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid rgba(255, 46, 77, 0.15);
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #ff2e4d !important;
        color: #ffffff !important;
        font-weight: 700;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================
# 3. PATHS & CONSTANTS
# =========================================================

IMG_SIZE = (224, 224)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "vgg16_pcb.keras"
CLASS_NAMES_PATH = ARTIFACTS_DIR / "class_names.json"


# =========================================================
# 4. CELL INFORMATION
# =========================================================

CELL_INFO = {
    "basophil": {
        "display": "Basophil",
        "category": "Granulocyte",
        "description": "Least common granulocyte. Key mediator of inflammatory and allergic responses; releases histamine and heparin.",
        "icon": "🧪",
        "normal_pct": "0.5% - 1.0%",
    },
    "eosinophil": {
        "display": "Eosinophil",
        "category": "Granulocyte",
        "description": "Granulocyte responsible for combating multicellular parasites and managing allergic reactions.",
        "icon": "🛡️",
        "normal_pct": "1.0% - 4.0%",
    },
    "erythroblast": {
        "display": "Erythroblast",
        "category": "Red Cell Precursor",
        "description": "Nucleated precursor to mature erythrocytes (red blood cells). Typically found in bone marrow.",
        "icon": "🩸",
        "normal_pct": "< 0.1% (In peripheral blood)",
    },
    "ig": {
        "display": "Immature Granulocyte (IG)",
        "category": "Granulocyte Precursor",
        "description": "Early-stage granulocytes such as myelocytes and metamyelocytes. Elevated in acute bacterial infections.",
        "icon": "⚡",
        "normal_pct": "< 0.5%",
    },
    "lymphocyte": {
        "display": "Lymphocyte",
        "category": "Agranulocyte",
        "description": "Primary cellular component of adaptive immunity (T-cells, B-cells, NK cells).",
        "icon": "🔬",
        "normal_pct": "20.0% - 40.0%",
    },
    "monocyte": {
        "display": "Monocyte",
        "category": "Agranulocyte",
        "description": "Largest white blood cell. Differentiates into tissue macrophages and dendritic cells for phagocytosis.",
        "icon": "🧫",
        "normal_pct": "2.0% - 8.0%",
    },
    "neutrophil": {
        "display": "Neutrophil",
        "category": "Granulocyte",
        "description": "Most abundant white blood cell. First responder to bacterial infections and acute tissue inflammation.",
        "icon": "⚔️",
        "normal_pct": "40.0% - 70.0%",
    },
    "platelet": {
        "display": "Platelet (Thrombocyte)",
        "category": "Cell Fragment",
        "description": "Small, non-nucleated cell fragments critical for primary blood coagulation and hemostasis.",
        "icon": "🩹",
        "normal_pct": "150,000 - 450,000 /µL",
    },
}


# =========================================================
# 5. MODEL & CLASS LOADING
# =========================================================

def _build_vgg16_transfer_model(num_classes: int = 8) -> keras.Model:
    data_augmentation = keras.Sequential(
        [
            keras.Input(shape=(224, 224, 3)),
            keras.layers.RandomFlip("horizontal", name="random_flip"),
            keras.layers.RandomRotation(0.05, fill_mode="reflect", name="random_rotation"),
            keras.layers.RandomZoom(height_factor=0.1, fill_mode="reflect", name="random_zoom"),
        ],
        name="sequential",
    )

    base_model = keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    model = keras.Sequential(
        [
            keras.Input(shape=(224, 224, 3), name="input_layer_2"),
            data_augmentation,
            keras.layers.Lambda(preprocess_input, name="lambda"),
            base_model,
            keras.layers.Flatten(name="flatten"),
            keras.layers.Dense(512, activation="relu", name="dense"),
            keras.layers.Dropout(0.5, name="dropout"),
            keras.layers.Dense(num_classes, activation="softmax", name="dense_1"),
        ],
        name="vgg16_transfer",
    )
    return model


@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        return None, None, f"Model file not found:\n{MODEL_PATH}"

    if not CLASS_NAMES_PATH.exists():
        return None, None, f"Class names file not found:\n{CLASS_NAMES_PATH}"

    try:
        model = _build_vgg16_transfer_model(num_classes=8)
        model.load_weights(str(MODEL_PATH))
    except Exception as e:
        return None, None, (
            "The model file was found, but its weights could not be loaded into the reconstructed architecture.\n\n"
            f"Error:\n{str(e)}"
        )

    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception as e:
        return None, None, (
            "The model loaded successfully, but class_names.json could not be read.\n\n"
            f"Error:\n{str(e)}"
        )

    if not isinstance(class_names, list):
        return None, None, "class_names.json must contain a JSON list of class names."

    if len(class_names) != 8:
        return None, None, f"Expected 8 class names, but found {len(class_names)}."

    return model, class_names, None


model, class_names, load_error = load_model_and_classes()


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.html("""
        <div style="text-align: center; padding: 10px 0 20px 0;">
            <div style="font-size: 3rem;">🩸</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #ffffff; margin-top: 6px;">
                HEMA<span style="color:#ff2e4d;">-AI</span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8;">
                Peripheral Blood Cell Classifier
            </div>
        </div>
    """)

    st.html('<div class="sidebar-section-title">Model Specifications</div>')

    st.markdown("""
    - **Architecture:** VGG16 (Transfer Learning)
    - **Input Shape:** 224 × 224 × 3 RGB
    - **Target Classes:** 8 Cell Types
    - **Pre-processing:** ImageNet Normalization
    """)

    st.html('<div class="sidebar-section-title">Target Cell Classes</div>')

    for c_key in CELL_INFO:
        info = CELL_INFO[c_key]
        st.markdown(f"- {info['icon']} **{info['display']}**")

    st.markdown("---")

    st.html("""
        <div style="
            background: rgba(255,46,77,0.1);
            border: 1px solid rgba(255,46,77,0.3);
            padding: 12px;
            border-radius: 10px;
            font-size: 0.78rem;
            color: #cbd5e1;
            text-align: center;
        ">
            ⚠️ <strong>Medical Disclaimer:</strong><br>
            This AI application is designed strictly for research & demonstration.<br>
            Not intended for direct diagnostic clinical use.
        </div>
    """)


# =========================================================
# 7. MAIN HEADER
# =========================================================

st.html("""
    <div class="hero-header">
        <div class="hero-title">
            🩸 <span>HEMA-AI</span> Blood Cell Analyzer
        </div>
        <div class="hero-subtitle">
            Upload a microscopic peripheral blood cell image
            to perform automated deep learning classification
            across 8 cellular categories using a pre-trained
            VGG16 convolutional neural network.
        </div>
    </div>
""")


# =========================================================
# 8. MODEL STATUS
# =========================================================

if model is None or class_names is None:
    st.error("⚠️ Model Loading Failed")
    if load_error:
        st.code(load_error)
    st.markdown(f"""
    **Expected model location:**  
    `{MODEL_PATH}`

    **Expected class names location:**  
    `{CLASS_NAMES_PATH}`
    """)
    st.stop()


# =========================================================
# 9. MAIN TABS
# =========================================================

tab_classify, tab_atlas, tab_about = st.tabs([
    "🔬 Image Classification",
    "📚 Cell Atlas & Info",
    "⚙️ System Details",
])


# =========================================================
# TAB 1 — IMAGE CLASSIFICATION
# =========================================================

with tab_classify:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    # -----------------------------------------------------
    # UPLOAD SECTION
    # -----------------------------------------------------
    with col_upload:
        st.html('<div class="custom-card">')
        st.subheader("1. Upload Microscopic Image")
        st.caption("Supported formats: JPEG, PNG (Single blood cell magnification recommended)")

        uploaded_file = st.file_uploader(
            "Drop your microscopic blood image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Blood Micrograph", use_container_width=True)

                w, h = image.size
                file_size_kb = len(uploaded_file.getvalue()) // 1024

                st.html(f"""
                    <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                        <div class="metric-card">
                            <div class="metric-val">{w}×{h}</div>
                            <div class="metric-lbl">Resolution</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val">RGB</div>
                            <div class="metric-lbl">Color Space</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-val">{file_size_kb} KB</div>
                            <div class="metric-lbl">File Size</div>
                        </div>
                    </div>
                """)

                analyze_btn = st.button("🔬 Run Cellular Analysis", type="primary")
            except Exception as e:
                st.error(f"Unable to read image: {e}")
                analyze_btn = False
        else:
            analyze_btn = False
            st.info("💡 Please upload an image above to begin cellular classification.")

        st.html("</div>")

    # -----------------------------------------------------
    # RESULT SECTION
    # -----------------------------------------------------
    with col_result:
        st.html('<div class="custom-card">')
        st.subheader("2. Classification Results")

        if uploaded_file is not None and analyze_btn:
            with st.spinner("Processing image and calculating probabilities..."):
                try:
                    img_resized = image.resize(IMG_SIZE)
                    img_array = np.array(img_resized).astype("float32")
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = model.predict(img_array, verbose=0)[0]

                    if len(preds) != len(class_names):
                        raise ValueError("Model output size does not match the number of class names.")

                    preds = np.asarray(preds, dtype=np.float32)
                    prediction_sum = np.sum(preds)
                    if prediction_sum > 0 and not np.isclose(prediction_sum, 1.0, atol=1e-3):
                        preds = preds / prediction_sum

                    top_idx = int(np.argmax(preds))
                    top_class_raw = str(class_names[top_idx])
                    top_conf = float(preds[top_idx]) * 100

                    info = CELL_INFO.get(top_class_raw.lower(), {
                        "display": top_class_raw,
                        "category": "Cell Type",
                        "description": "Classified cellular object.",
                        "icon": "🔬",
                        "normal_pct": "N/A",
                    })
                except Exception as e:
                    st.error("Prediction failed.")
                    st.code(str(e))
                    st.stop()

            # Top Result
            st.html(f"""
                <div class="result-box">
                    <div class="result-badge">{info['category']}</div>
                    <div style="font-size: 2.5rem; margin-top: 4px;">{info['icon']}</div>
                    <div class="result-class">{info['display']}</div>
                    <div class="result-confidence">Confidence: {top_conf:.2f}%</div>
                </div>
            """)

            st.markdown(f"**Cell Description:** {info['description']}")
            st.markdown(f"**Typical Reference Range:** `{info['normal_pct']}`")
            st.markdown("---")

            # Probability Distribution
            st.markdown("#### Probability Distribution")

            sorted_indices = np.argsort(preds)[::-1]

            for idx in sorted_indices:
                cls_raw = str(class_names[idx])
                prob_pct = float(preds[idx]) * 100
                cls_display = CELL_INFO.get(cls_raw.lower(), {}).get("display", cls_raw.capitalize())
                is_top = idx == top_idx
                fill_class = "prob-fill prob-fill-top" if is_top else "prob-fill"
                bar_width = max(prob_pct, 1.5)

                st.html(f"""
                    <div class="prob-bar-container">
                        <div class="prob-header">
                            <span class="prob-name">{"🔥 " if is_top else ""}{cls_display}</span>
                            <span class="prob-val">{prob_pct:.2f}%</span>
                        </div>
                        <div class="prob-track">
                            <div class="{fill_class}" style="width: {bar_width:.2f}%;"></div>
                        </div>
                    </div>
                """)

        elif uploaded_file is not None and not analyze_btn:
            st.info("👈 Click **Run Cellular Analysis** to analyze the uploaded image.")
        else:
            st.html("""
                <div style="text-align: center; padding: 40px 20px; color: #64748b;">
                    <div style="font-size: 3rem; margin-bottom: 10px;">🧪</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">Awaiting Input Image</div>
                    <div style="font-size: 0.85rem;">
                        Upload a cell micrograph on the left pane to run neural network evaluation.
                    </div>
                </div>
            """)

        st.html("</div>")


# =========================================================
# TAB 2 — CELL ATLAS
# =========================================================

with tab_atlas:
    st.subheader("📚 Peripheral Blood Cell Reference Atlas")
    st.write("Detailed guide on the 8 cellular types recognized by the VGG16 model:")

    grid_cols = st.columns(2, gap="medium")
    cell_keys = list(CELL_INFO.keys())

    for idx, c_key in enumerate(cell_keys):
        col_target = grid_cols[idx % 2]
        c_data = CELL_INFO[c_key]

        with col_target:
            st.html(f"""
                <div class="custom-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="font-size: 1.4rem; font-weight: 700; color: #ffffff;">
                            {c_data['icon']} {c_data['display']}
                        </div>
                        <span class="result-badge">{c_data['category']}</span>
                    </div>
                    <p style="color: #cbd5e1; margin-top: 10px; font-size: 0.92rem; line-height: 1.5;">
                        {c_data['description']}
                    </p>
                    <div style="font-size: 0.85rem; color: #ff4d6d; font-family: 'JetBrains Mono', monospace; margin-top: 8px;">
                        Reference Range: {c_data['normal_pct']}
                    </div>
                </div>
            """)


# =========================================================
# TAB 3 — SYSTEM DETAILS
# =========================================================

with tab_about:
    st.subheader("⚙️ VGG16 Neural Network Architecture")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.html("""
            <div class="custom-card">
                <h4 style="color: #ff2e4d; margin-top:0;">Model Pipeline Summary</h4>
                <ul>
                    <li><b>Base Model:</b> VGG16 initialized with ImageNet pre-trained weights.</li>
                    <li><b>Input Preprocessing:</b> ImageNet VGG16 <code>preprocess_input</code> normalization.</li>
                    <li><b>Transfer Learning:</b> VGG16 feature extraction layers were trained/fine-tuned on the PBC dataset.</li>
                    <li><b>Classifier Head:</b> Dense layers followed by Softmax activation producing 8 class probabilities.</li>
                    <li><b>Input:</b> 224 × 224 × 3 RGB image.</li>
                    <li><b>Output:</b> 8 blood cell categories.</li>
                </ul>
            </div>
        """)

    with col_b:
        st.html(f"""
            <div class="custom-card">
                <h4 style="color: #ff2e4d; margin-top:0;">Artifact Paths & Status</h4>
                <p><b>Model File:</b> <code>vgg16_pcb.keras</code></p>
                <p><b>Classes File:</b> <code>class_names.json</code></p>
                <p><b>Active Artifact Path:</b> <code>{ARTIFACTS_DIR}</code></p>
                <p>
                    <b>Model Status:</b>
                    <span style="color: #22c55e; font-weight: 700;">ACTIVE & READY</span>
                </p>
                <p><b>Model Input:</b> <code>224 × 224 × 3</code></p>
                <p><b>Number of Classes:</b> <code>{len(class_names)}</code></p>
            </div>
        """)


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("---")
st.html("""
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 10px 0;">
        🩸 <strong>HEMA-AI Peripheral Blood Cell Classifier</strong>
        | Powered by Streamlit & TensorFlow
        | Black & Red Edition
    </div>
""")