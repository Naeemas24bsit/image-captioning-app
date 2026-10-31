import streamlit as st
import torch
import hashlib
import html
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VisionAI | AI Image Captioning",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "naeemajalal/image-captioning-blip-flickr8k"

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 15% 10%, #252052 0%, transparent 38%),
        radial-gradient(ellipse at 90% 85%, #102e46 0%, transparent 40%),
        #080c18;
    color: #f8fafc;
}

header, footer, #MainMenu {
    visibility: hidden;
}

.block-container {
    max-width: 1380px;
    padding: 1rem 2.5rem 1rem;
}

[data-testid="stSidebar"] {
    display: none;
}

.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0 18px;
    border-bottom: 1px solid rgba(148,163,184,.15);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 21px;
    font-weight: 800;
    color: white;
    letter-spacing: -.7px;
}

.brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg,#8b5cf6,#2563eb);
    font-size: 21px;
    box-shadow: 0 5px 25px #6246bb55;
}

.nav-right {
    color: #a5b4cf;
    font-size: 12px;
    font-weight: 600;
}

.status-dot {
    display: inline-block;
    height: 8px;
    width: 8px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 7px;
    box-shadow: 0 0 10px #34d399;
}

.hero {
    text-align: center;
    padding: 24px 0 22px;
}

.eyebrow {
    display: inline-block;
    color: #c4b5fd;
    background: rgba(124,58,237,.15);
    border: 1px solid rgba(167,139,250,.3);
    padding: 7px 15px;
    border-radius: 30px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.hero h1 {
    font-size: clamp(34px, 4vw, 49px);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -2px;
    margin: 13px 0 7px;
    background: linear-gradient(90deg,#fff,#b9a6ff,#67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #94a3b8;
    font-size: 13px;
    margin: 0;
}

.section-title {
    color: #f1f5f9;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #8795ad;
    font-size: 11px;
    margin-bottom: 15px;
}

.panel {
    background: linear-gradient(
        145deg,
        rgba(30,41,70,.85),
        rgba(15,23,42,.8)
    );
    border: 1px solid rgba(148,163,184,.17);
    border-radius: 19px;
    padding: 20px;
    min-height: 390px;
    box-shadow: 0 15px 45px rgba(0,0,0,.18);
}

.upload-hint {
    color: #a5b4cf;
    font-size: 12px;
    text-align: center;
    margin: 8px 0;
}

[data-testid="stFileUploader"] {
    background: rgba(8,15,31,.65);
    border: 1px dashed #7564c7;
    border-radius: 13px;
    padding: 10px;
}

[data-testid="stFileUploader"] section {
    padding: 5px;
}

[data-testid="stImage"] img {
    max-height: 285px;
    object-fit: contain;
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    min-height: 47px;
    border: 1px solid transparent;
    border-radius: 12px;
    color: white;
    background: linear-gradient(100deg,#7c3aed,#315eea);
    font-weight: 700;
    font-size: 13px;
    box-shadow: 0 7px 25px rgba(91,67,210,.25);
    transition: all .2s ease;
}

.stButton > button:hover {
    color: white;
    border: 1px solid #a78bfa;
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(91,67,210,.4);
}

.result-empty {
    height: 270px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    border: 1px dashed rgba(148,163,184,.25);
    border-radius: 15px;
    background: rgba(8,15,31,.35);
    padding: 20px;
}

.empty-icon {
    font-size: 38px;
    margin-bottom: 15px;
    opacity: .8;
}

.empty-title {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 15px;
}

.empty-text {
    color: #8795ad;
    font-size: 11px;
    margin-top: 8px;
    line-height: 1.7;
}

.caption-card {
    background: linear-gradient(135deg,rgba(88,54,160,.23),rgba(21,54,83,.45));
    border: 1px solid rgba(139,117,235,.42);
    border-radius: 16px;
    padding: 23px;
    margin-top: 12px;
    min-height: 180px;
}

.caption-label {
    color: #b9a6ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 15px;
}

.caption-text {
    color: #fff;
    font-size: 19px;
    font-weight: 600;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.model-info {
    color: #8998b2;
    font-size: 10px;
    margin-top: 17px;
    padding-top: 13px;
    border-top: 1px solid rgba(148,163,184,.15);
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 10px;
    padding: 18px 0 0;
}

@media(max-width: 800px) {
    .block-container {
        padding: .8rem 1rem;
    }
    .panel {
        min-height: auto;
        padding: 16px;
    }
    .hero {
        padding: 22px 0;
    }
    .caption-text {
        font-size: 16px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    return processor, model, device


# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
    <div class="brand">
        <div class="brand-icon">✦</div>
        VisionAI
    </div>
    <div class="nav-right">
        <span class="status-dot"></span> AI IMAGE ENGINE
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <span class="eyebrow">DEEP LEARNING · TRANSFER LEARNING</span>
    <h1>See Beyond the Image.</h1>
    <p>Upload an image. Let artificial intelligence turn it into words.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
try:
    with st.spinner("Initializing VisionAI engine..."):
        processor, model, device = load_model()

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# ---------------- MAIN DASHBOARD ----------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <div class="section-title">01 / Image Workspace</div>
        <div class="section-subtitle">
            Upload your image to begin the analysis
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label="Upload image"
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

        st.markdown(
            f'<p class="upload-hint">{html.escape(uploaded_file.name)}</p>',
            unsafe_allow_html=True
        )

    else:
        st.markdown("""
        <div class="upload-hint">
            JPG, JPEG or PNG · Select an image to continue
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown("""
    <div class="panel">
        <div class="section-title">02 / AI Caption Studio</div>
        <div class="section-subtitle">
            Your image description will appear here
        </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        generate = st.button(
            "✦  Generate Image Caption",
            use_container_width=True
        )

        if generate:
            try:
                with st.spinner("VisionAI is understanding your image..."):
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
                st.error(f"Caption generation failed: {e}")

        if st.session_state.get("caption"):
            safe_caption = html.escape(
                st.session_state["caption"]
            )

            st.markdown(f"""
            <div class="caption-card">
                <div class="caption-label">
                    ✦ GENERATED DESCRIPTION
                </div>
                <div class="caption-text">{safe_caption}</div>
                <div class="model-info">
                    MODEL: FINE-TUNED BLIP · FLICKR8K
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="result-empty">
                <div class="empty-icon">✧</div>
                <div class="empty-title">Ready to understand your image</div>
                <div class="empty-text">
                    Click Generate Image Caption.<br>
                    Your AI-generated description will appear here.
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-empty">
            <div class="empty-icon">✧</div>
            <div class="empty-title">Your caption awaits</div>
            <div class="empty-text">
                Upload an image in the workspace<br>
                to activate the AI caption generator.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    VISIONAI &nbsp;·&nbsp; POWERED BY BLIP &nbsp;·&nbsp;
    DEEP LEARNING PROJECT
</div>
""", unsafe_allow_html=True)
