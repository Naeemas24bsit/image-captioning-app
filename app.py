import streamlit as st
import torch
import hashlib
import html
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="VisionAI | Image Captioning",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "naeemajalal/image-captioning-blip-flickr8k"

# ---------------- PREMIUM COMPACT DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 10%, #252052 0%, transparent 38%),
        radial-gradient(ellipse at 90% 90%, #102e46 0%, transparent 40%),
        #080c18;
    color: #f8fafc;
}

header, footer, #MainMenu {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1380px;
    padding: 0.7rem 2rem 0.5rem;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0 12px;
    border-bottom: 1px solid rgba(148,163,184,.16);
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 20px;
    font-weight: 800;
    color: white;
}

.brand-icon {
    width: 34px;
    height: 34px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#8b5cf6,#2563eb);
    font-size: 20px;
    box-shadow: 0 5px 20px #6246bb55;
}

.nav-status {
    color: #a5b4cf;
    font-size: 11px;
    font-weight: 600;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34d399;
    margin-right: 6px;
    box-shadow: 0 0 9px #34d399;
}

.hero {
    text-align: center;
    padding: 17px 0 17px;
}

.eyebrow {
    display: inline-block;
    color: #c4b5fd;
    background: rgba(124,58,237,.15);
    border: 1px solid rgba(167,139,250,.3);
    padding: 5px 13px;
    border-radius: 30px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.4px;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 44px);
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -1.8px;
    margin: 10px 0 5px;
    background: linear-gradient(90deg,#fff,#b9a6ff,#67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 12px;
    margin: 0;
}

.section-heading {
    color: #f1f5f9;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 3px;
}

.section-desc {
    color: #8795ad;
    font-size: 10px;
    margin-bottom: 12px;
}

.panel {
    background: linear-gradient(
        145deg,
        rgba(30,41,70,.88),
        rgba(15,23,42,.88)
    );
    border: 1px solid rgba(148,163,184,.19);
    border-radius: 18px;
    padding: 17px;
    min-height: 350px;
    box-shadow: 0 12px 35px rgba(0,0,0,.18);
}

[data-testid="stFileUploader"] {
    background: rgba(8,15,31,.65);
    border: 1px dashed #7564c7;
    border-radius: 12px;
    padding: 7px;
}

[data-testid="stFileUploader"] section {
    padding: 3px;
}

[data-testid="stImage"] img {
    max-height: 235px;
    object-fit: contain;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    min-height: 43px;
    border: 1px solid transparent;
    border-radius: 11px;
    color: white;
    background: linear-gradient(100deg,#7c3aed,#315eea);
    font-weight: 700;
    font-size: 12px;
    box-shadow: 0 6px 20px rgba(91,67,210,.25);
    transition: all .2s ease;
}

.stButton > button:hover {
    color: white;
    border: 1px solid #a78bfa;
    transform: translateY(-2px);
    background: linear-gradient(100deg,#6d28d9,#1d4ed8);
}

.result-empty {
    min-height: 245px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    border: 1px dashed rgba(148,163,184,.25);
    border-radius: 14px;
    background: rgba(8,15,31,.35);
    padding: 15px;
}

.empty-icon {
    font-size: 34px;
    margin-bottom: 12px;
    color: #a78bfa;
}

.empty-title {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 14px;
}

.empty-text {
    color: #8795ad;
    font-size: 10px;
    margin-top: 7px;
    line-height: 1.6;
}

.caption-card {
    background: linear-gradient(135deg,rgba(88,54,160,.25),rgba(21,54,83,.5));
    border: 1px solid rgba(139,117,235,.42);
    border-radius: 15px;
    padding: 20px;
    margin-top: 8px;
    min-height: 175px;
}

.caption-label {
    color: #b9a6ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.3px;
    margin-bottom: 13px;
}

.caption-text {
    color: white;
    font-size: 18px;
    font-weight: 600;
    line-height: 1.55;
    overflow-wrap: anywhere;
}

.model-info {
    color: #8998b2;
    font-size: 9px;
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid rgba(148,163,184,.15);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 9px;
    padding: 12px 0 0;
}

/* Compact desktop: avoid page-level scrolling */
@media (min-width: 900px) and (min-height: 680px) {
    .stApp {
        height: 100vh;
        overflow-y: hidden;
    }

    .main .block-container {
        height: 100vh;
        overflow-y: hidden;
    }
}

/* Smaller screens: allow natural scrolling */
@media (max-width: 899px), (max-height: 679px) {
    .block-container {
        padding: .7rem 1rem 1.5rem;
    }

    .panel {
        min-height: auto;
        padding: 15px;
    }

    .hero {
        padding: 17px 0;
    }

    .caption-text {
        font-size: 16px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
    <div class="brand">
        <div class="brand-icon">✦</div>
        VisionAI
    </div>
    <div class="nav-status">
        <span class="status-dot"></span> AI ENGINE ONLINE
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <span class="eyebrow">DEEP LEARNING · TRANSFER LEARNING</span>
    <h1>See Beyond the Image.</h1>
    <p>Transform your images into meaningful words with AI.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
try:
    with st.spinner("Initializing VisionAI..."):
        processor, model, device = load_model()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- DASHBOARD ----------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="section-heading">01 / IMAGE WORKSPACE</div>
        <div class="section-desc">
            Upload an image to start
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image_id = hashlib.sha256(image_bytes).hexdigest()

        if st.session_state.get("image_id") != image_id:
            st.session_state["image_id"] = image_id
            st.session_state["caption"] = None

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(uploaded_file.name)

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown("""
    <div class="panel">
        <div class="section-heading">02 / AI CAPTION STUDIO</div>
        <div class="section-desc">
            Your intelligent image description
        </div>
    """, unsafe_allow_html=True)

    if uploaded_file:

        if st.button("✦  GENERATE IMAGE CAPTION"):
            try:
                with st.spinner("Analyzing your image..."):

                    inputs = processor(
                        images=image,
                        return_tensors="pt"
                    ).to(device)

                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=40
                        )

                    caption = processor.decode(
                        output[0],
                        skip_special_tokens=True
                    )

                    st.session_state["caption"] = caption

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        if st.session_state.get("caption"):
            safe_caption = html.escape(
                st.session_state["caption"]
            )

            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-label">
                    ✦ AI GENERATED DESCRIPTION
                </div>
                <div class="caption-text">{safe_caption}</div>
                <div class="model-info">
                    FINE-TUNED BLIP · FLICKR8K DATASET
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-empty">
                <div class="empty-icon">✧</div>
                <div class="empty-title">Your caption awaits</div>
                <div class="empty-text">
                    Click Generate Image Caption<br>
                    to see your AI-generated result.
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-empty">
            <div class="empty-icon">✧</div>
            <div class="empty-title">Ready to analyze</div>
            <div class="empty-text">
                Upload an image to activate<br>
                the caption generator.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    VISIONAI · POWERED BY FINE-TUNED BLIP
</div>
""", unsafe_allow_html=True)
