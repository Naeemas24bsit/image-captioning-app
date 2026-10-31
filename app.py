import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="VisionAI | Image Captioning",
    page_icon="✨",
    layout="wide"
)

MODEL_ID = "naeemajalal/image-captioning-blip-flickr8k"

# Custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #090d1b, #11162b 60%, #17132c);
    color: #f1f5f9;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    text-align: center;
    padding: 35px 15px 28px;
}

.tag {
    display: inline-block;
    padding: 8px 16px;
    border: 1px solid #6654a9;
    border-radius: 30px;
    background: #211b40;
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: clamp(36px, 6vw, 58px);
    font-weight: 800;
    margin: 20px 0 10px;
    background: linear-gradient(90deg, #ffffff, #a78bfa, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a5b4cf;
    font-size: 16px;
}

.panel {
    background: rgba(30, 41, 70, 0.65);
    border: 1px solid rgba(148, 163, 184, 0.19);
    border-radius: 20px;
    padding: 25px;
    min-height: 230px;
}

.panel h3 {
    color: #e2e8f0;
    font-size: 19px;
}

.muted {
    color: #94a3b8;
    font-size: 13px;
}

.result {
    background: linear-gradient(135deg, #251d49, #172c46);
    border: 1px solid #7160c5;
    border-radius: 18px;
    padding: 25px;
    margin-top: 22px;
}

.result-label {
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.result-text {
    color: #ffffff;
    font-size: 21px;
    line-height: 1.6;
    font-weight: 600;
    margin-top: 12px;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 15px;
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white;
    font-weight: 700;
    font-size: 16px;
    transition: 0.25s;
}

.stButton > button:hover {
    border: none;
    background: linear-gradient(90deg, #6d28d9, #1d4ed8);
    transform: translateY(-2px);
    color: white;
}

[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.7);
    border: 1px dashed #7463c6;
    border-radius: 15px;
    padding: 15px;
}

hr {
    border-color: #29344f;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 30px 0 10px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


# Header
st.markdown("""
<div class="hero">
    <span class="tag">✦ DEEP LEARNING PROJECT</span>
    <h1>VisionAI</h1>
    <p>Transform your images into meaningful words with Artificial Intelligence.</p>
</div>
""", unsafe_allow_html=True)

# Model loading
try:
    with st.spinner("Initializing VisionAI model..."):
        processor, model = load_model()
    st.success("✦ AI model is ready")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.markdown("---")

# Main layout
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="panel">
        <h3>📤 Upload Your Image</h3>
        <p class="muted">Choose an image and let AI understand its content.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Your uploaded image",
            use_container_width=True
        )

with right:
    st.markdown("""
    <div class="panel">
        <h3>🧠 AI Image Understanding</h3>
        <p class="muted">
        Our fine-tuned BLIP model analyzes the uploaded image
        and generates a natural language caption.
        </p>
        <p style="color:#a78bfa;font-weight:600;">
        Powered by Deep Learning & Transfer Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file:
        if st.button("✦ Generate Image Caption"):
            try:
                with st.spinner("AI is analyzing your image..."):
                    inputs = processor(
                        images=image,
                        return_tensors="pt"
                    )

                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

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
            st.markdown(f"""
            <div class="result">
                <div class="result-label">✦ AI GENERATED CAPTION</div>
                <div class="result-text">{st.session_state["caption"]}</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Upload an image to start generating captions.")

st.markdown("""
<div class="footer">
    VisionAI · Image Captioning System · Powered by BLIP
    <br><br>
    Developed as a Deep Learning Project
</div>
""", unsafe_allow_html=True)
