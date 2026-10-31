import streamlit as st
import torch
import hashlib
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vision AI | Image Captioning",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_ID = "naeemajalal/image-captioning-blip-flickr8k"

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

    model.to(device)
    model.eval()

    return processor, model, device


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: #080b16;
    color: #ffffff;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    display: none;
}

[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1450px;
    padding: 1rem 2.5rem 0.5rem;
}

[data-testid="stAppViewContainer"] {
    height: 100vh;
    overflow: hidden;
}

[data-testid="stMain"] {
    height: calc(100vh - 10px);
    overflow: hidden;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
    background: linear-gradient(90deg, #ffffff, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #929bb5;
    font-size: 13px;
    margin-top: 5px;
}

.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0 15px;
    border-bottom: 1px solid #222941;
    margin-bottom: 22px;
}

.brand {
    font-size: 19px;
    font-weight: 800;
    color: #ffffff;
}

.brand span {
    color: #a78bfa;
}

.status {
    background: #142c29;
    border: 1px solid #235c4d;
    color: #66e0b5;
    border-radius: 20px;
    padding: 6px 13px;
    font-size: 11px;
}

.hero {
    text-align: center;
    margin-bottom: 22px;
}

.panel {
    background: linear-gradient(145deg, #151b30, #101526);
    border: 1px solid #28314d;
    border-radius: 18px;
    padding: 20px;
    height: 410px;
    box-shadow: 0 10px 35px rgba(0,0,0,0.18);
}

.panel-title {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 5px;
}

.panel-subtitle {
    color: #929bb5;
    font-size: 12px;
    margin-bottom: 17px;
}

.caption-placeholder {
    background: #0b1020;
    border: 1px solid #28314d;
    border-radius: 13px;
    padding: 20px;
    min-height: 130px;
    color: #aab3ca;
    font-size: 14px;
    line-height: 1.7;
}

.caption-result {
    background: linear-gradient(135deg, #171e36, #101a2e);
    border: 1px solid #6250a8;
    border-radius: 13px;
    padding: 20px;
    min-height: 130px;
    color: #f0edff;
    font-size: 15px;
    line-height: 1.8;
}

.caption-label {
    color: #a78bfa;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

div.stButton > button {
    width: 100%;
    height: 47px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(100deg, #7655e8, #3978ec);
    color: white;
    font-size: 14px;
    font-weight: 700;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    border: none;
    transform: translateY(-2px);
    box-shadow: 0 5px 20px #6549cf55;
    color: white;
}

div.stButton > button:disabled {
    background: #30374c;
    color: #929bb5;
}

[data-testid="stFileUploader"] {
    background: #0b1020;
    border: 1px dashed #7560c5;
    border-radius: 13px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] button {
    background: #292348;
    color: white;
    border: 1px solid #7560c5;
    border-radius: 9px;
}

[data-testid="stFileUploader"] small {
    color: #929bb5;
}

[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

.footer {
    text-align: center;
    color: #66718e;
    font-size: 11px;
    padding-top: 14px;
}

@media (max-width: 900px) {
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        height: auto;
        min-height: 100vh;
        overflow: auto;
    }

    .block-container {
        padding: 1rem;
    }

    .panel {
        height: auto;
        min-height: 350px;
    }

    .main-title {
        font-size: 27px;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------- NAVBAR ----------------
st.markdown("""
<div class="navbar">
    <div class="brand">VISION<span>AI</span> ✦</div>
    <div class="status">● AI MODEL ONLINE</div>
</div>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <p class="main-title">Image Intelligence Studio</p>
    <p class="subtitle">
        Transform your images into meaningful descriptions with AI
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE ----------------
if "caption" not in st.session_state:
    st.session_state.caption = ""

if "image_hash" not in st.session_state:
    st.session_state.image_hash = ""


# ---------------- TWO MAIN SECTIONS ----------------
left, right = st.columns([1, 1], gap="large")


# LEFT: CAPTION STUDIO
with left:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">✦ Caption Studio</div>
        <div class="panel-subtitle">
            AI-generated description of your image
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.caption:
        st.markdown(
            f'<div class="caption-label">GENERATED CAPTION</div>'
            f'<div class="caption-result">{st.session_state.caption}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class="caption-label">YOUR CAPTION WILL APPEAR HERE</div>
        <div class="caption-placeholder">
            Upload an image on the right and click Generate Caption
            to see the AI-generated description.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    generate = st.button(
        "✦  Generate Caption",
        use_container_width=True,
        disabled="uploaded_image" not in st.session_state
    )

    st.markdown("</div>", unsafe_allow_html=True)


# RIGHT: IMAGE UPLOAD
with right:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">▧ Upload Image</div>
        <div class="panel-subtitle">
            Select an image for AI analysis
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose your image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

        if image_hash != st.session_state.image_hash:
            st.session_state.image_hash = image_hash
            st.session_state.caption = ""
            st.session_state.uploaded_image = image

        st.image(image, width=320)

    else:
        st.session_state.pop("uploaded_image", None)
        st.session_state.image_hash = ""
        st.session_state.caption = ""

        st.markdown("""
        <div style="
            height:180px;
            display:flex;
            justify-content:center;
            align-items:center;
            background:#0b1020;
            border:1px dashed #3c4562;
            border-radius:13px;
            color:#77829e;
            font-size:13px;">
            Your image preview will appear here
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- GENERATE CAPTION ----------------
if generate and "uploaded_image" in st.session_state:

    try:
        with st.spinner("AI is analyzing your image..."):

            processor, model, device = load_model()

            inputs = processor(
                images=st.session_state.uploaded_image,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=5
                )

            caption = processor.decode(
                output[0],
                skip_special_tokens=True
            )

            st.session_state.caption = caption

        st.rerun()

    except Exception as e:
        st.error(f"Caption generation failed: {e}")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    POWERED BY FINE-TUNED BLIP MODEL · FLICKR8K DATASET
</div>
""", unsafe_allow_html=True)
