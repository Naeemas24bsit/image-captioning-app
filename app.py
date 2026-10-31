import hashlib
import html
import time

import streamlit as st
import torch
from PIL import Image, ImageOps
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="VisionAI | Image Captioning",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_ID = "naeemajalal/image-captioning-blip-flickr8k"

# ---------------- STYLE ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&display=swap');

:root {
    --bg: #080a16;
    --surface: #10132b;
    --surface-2: #171b3a;
    --line: #2a3066;
    --text: #eef0ff;
    --muted: #9aa1cc;
    --accent: #8b7bff;
    --accent-2: #ff6fa5;
    --accent-3: #5eead4;
    --grad: linear-gradient(115deg, #6d5dfc 0%, #b04fd8 52%, #ff6b9d 100%);
    --ok: #5eead4;
}

.stApp {
    font-family: 'Manrope', sans-serif;
    background:
        radial-gradient(900px 520px at -5% -10%, rgba(109,93,252,.22), transparent 60%),
        radial-gradient(800px 520px at 105% 110%, rgba(255,107,157,.14), transparent 60%),
        radial-gradient(600px 400px at 100% -5%, rgba(94,234,212,.08), transparent 60%),
        var(--bg);
    color: var(--text);
}

.stButton > button,
[data-testid="stFileUploader"] button {
    font-family: 'Manrope', sans-serif;
}

/* hide Streamlit chrome */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
footer, #MainMenu, [data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

.block-container {
    max-width: 1320px;
    padding: 1rem 2rem 1rem !important;
}

/* ---------- top bar ---------- */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 44px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 4px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand-mark {
    width: 32px; height: 32px; border-radius: 9px;
    display: grid; place-items: center;
    background: var(--grad); color: #fff;
    font-size: 18px; font-weight: 800;
    box-shadow: 0 6px 18px rgba(109,93,252,.35);
}
.brand-name { font-size: 18px; font-weight: 800; letter-spacing: -.3px; }
.brand-sub { color: var(--muted); font-size: 12.5px; font-weight: 500; margin-left: 4px; }
.status {
    display: flex; align-items: center; gap: 8px;
    color: var(--muted); font-size: 12.5px; font-weight: 600;
    padding: 6px 12px; border: 1px solid var(--line); border-radius: 999px;
    background: rgba(23,27,58,.6);
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 8px var(--ok); }

/* ---------- panels (st.container(border=True)) ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(23,27,62,.92), rgba(14,17,40,.92));
    border: 1px solid rgba(139,123,255,.24) !important;
    border-radius: 18px !important;
    padding: 6px 10px;
    height: calc(100vh - 118px);
    overflow: hidden;
    box-shadow: 0 18px 40px rgba(0,0,0,.28);
}

.panel-title { font-size: 16px; font-weight: 700; margin: 4px 0 2px; }
.panel-desc { font-size: 13px; color: var(--muted); margin-bottom: 8px; }

/* ---------- uploader ---------- */
[data-testid="stFileUploader"] label { display: none; }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(23,27,58,.7);
    border: 1.5px dashed #3d4590;
    border-radius: 14px;
    padding: 12px 16px;
    transition: border-color .2s ease, background .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent);
    background: rgba(30,35,76,.8);
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] button:focus,
[data-testid="stFileUploaderDropzone"] button:active {
    background: var(--grad) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(109,93,252,.25);
}
[data-testid="stFileUploaderDropzone"] small { color: var(--muted); }

/* ---------- viewfinder preview ---------- */
.frame, [data-testid="stImage"] {
    --c: var(--accent-3);
    background:
        linear-gradient(var(--c), var(--c)) top left / 22px 2px no-repeat,
        linear-gradient(var(--c), var(--c)) top left / 2px 22px no-repeat,
        linear-gradient(var(--c), var(--c)) top right / 22px 2px no-repeat,
        linear-gradient(var(--c), var(--c)) top right / 2px 22px no-repeat,
        linear-gradient(var(--c), var(--c)) bottom left / 22px 2px no-repeat,
        linear-gradient(var(--c), var(--c)) bottom left / 2px 22px no-repeat,
        linear-gradient(var(--c), var(--c)) bottom right / 22px 2px no-repeat,
        linear-gradient(var(--c), var(--c)) bottom right / 2px 22px no-repeat;
    padding: 12px;
    display: flex;
    justify-content: center;
    align-items: center;
}
[data-testid="stImage"] img {
    width: auto !important;
    max-width: 100%;
    max-height: calc(100vh - 400px);
    min-height: 120px;
    object-fit: contain;
    border-radius: 8px;
}
[data-testid="stImageCaption"], [data-testid="stCaptionContainer"] {
    color: var(--muted);
    font-size: 12px;
    text-align: center;
}
.frame {
    height: calc(100vh - 400px);
    min-height: 160px;
    flex-direction: column;
    gap: 8px;
    text-align: center;
}
.frame-title { font-weight: 700; font-size: 15px; }
.frame-text { color: var(--muted); font-size: 13px; }

/* ---------- controls ---------- */
.stButton > button,
.stButton > button:hover,
.stButton > button:focus:not(:active),
.stButton > button:active {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: none !important;
    background: var(--grad) !important;
    background-size: 140% 100% !important;
    color: #fff !important;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: .1px;
    box-shadow: 0 8px 22px rgba(109,93,252,.24);
    transition: transform .15s ease, background-position .3s ease;
}
.stButton > button { background-position: 0% 0 !important; }
.stButton > button:hover {
    background-position: 100% 0 !important;
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus-visible { outline: 2px solid var(--accent-3); outline-offset: 2px; }
.stButton > button:disabled,
.stButton > button:disabled:hover {
    background: var(--surface-2) !important;
    color: var(--muted) !important;
    box-shadow: none;
    border: 1px solid var(--line) !important;
    transform: none;
}

[data-testid="stRadio"] label p { color: var(--muted); font-size: 13px; font-weight: 600; }
[data-testid="stRadio"] [role="radiogroup"] { gap: 6px; }

/* ---------- result ---------- */
.state {
    height: calc(100vh - 330px);
    min-height: 200px;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    text-align: center; gap: 8px;
    border: 1.5px dashed var(--line); border-radius: 14px;
    background: rgba(23,27,58,.55);
    padding: 16px;
}
.state-icon { font-size: 30px; color: #b79cff; }
.state-title { font-weight: 700; font-size: 15px; }
.state-text { color: var(--muted); font-size: 13px; line-height: 1.6; }

.result {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(109,93,252,.16), rgba(255,107,157,.08));
    border: 1px solid rgba(139,123,255,.28);
    border-radius: 14px;
    padding: 18px 22px 18px 26px;
    margin: 4px 0 8px;
    animation: reveal .5s ease both;
}
.result::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #6d5dfc, #ff6b9d);
}
.result-label { color: var(--muted); font-size: 12.5px; font-weight: 600; margin-bottom: 8px; }
.result-text {
    font-family: 'Fraunces', Georgia, serif;
    font-size: clamp(22px, 2.3vw, 30px);
    font-weight: 500;
    line-height: 1.35;
    letter-spacing: -.2px;
    color: #fff;
    overflow-wrap: anywhere;
    text-transform: capitalize;
}
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.chip {
    font-size: 12px; font-weight: 600; color: #c6caff;
    background: rgba(139,123,255,.1);
    border: 1px solid rgba(139,123,255,.28);
    border-radius: 999px; padding: 4px 10px;
}
@keyframes reveal {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: none; }
}
@media (prefers-reduced-motion: reduce) { .result { animation: none; } }

[data-testid="stCode"] pre { background: var(--surface-2) !important; border: 1px solid var(--line); border-radius: 10px; }

/* ---------- single screen (no page scroll) on desktop ---------- */
@media (min-width: 900px) and (min-height: 700px) {
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], section.main {
        height: 100vh;
        overflow: hidden !important;
    }
}

/* small screens: allow natural scrolling */
@media (max-width: 899px), (max-height: 699px) {
    .block-container { padding: .8rem 1rem 1.5rem !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { height: auto; overflow: visible; }
    .frame, .state { height: auto; min-height: 220px; }
    [data-testid="stImage"] img { max-height: 320px; }
    .brand-sub { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


try:
    with st.spinner("Loading model…"):
        processor, model, device = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# ---------------- TOP BAR ----------------
st.markdown(
    f"""
<div class="topbar">
    <div class="brand">
        <div class="brand-mark">◐</div>
        <span class="brand-name">VisionAI</span>
        <span class="brand-sub">Image captioning with fine-tuned BLIP</span>
    </div>
    <div class="status"><span class="dot"></span>Model ready · {device.type.upper()}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- DASHBOARD ----------------
left, right = st.columns([1, 1], gap="medium")
image = None
uploaded_file = None

# ----- left: image -----
with left:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Your image</div>'
            '<div class="panel-desc">JPG or PNG. Drop a file or browse.</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            image_id = hashlib.sha256(image_bytes).hexdigest()

            if st.session_state.get("image_id") != image_id:
                st.session_state["image_id"] = image_id
                st.session_state["caption"] = None
                st.session_state["meta"] = None

            image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
            st.image(image)
        else:
            st.session_state["image_id"] = None
            st.session_state["caption"] = None
            st.markdown(
                """
                <div class="frame">
                    <div class="frame-title">No image yet</div>
                    <div class="frame-text">Your photo will appear here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----- right: caption -----
with right:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title">Caption</div>'
            '<div class="panel-desc">A one-line description of what the model sees.</div>',
            unsafe_allow_html=True,
        )

        mode = st.radio(
            "Mode",
            ["Quick", "Accurate"],
            horizontal=True,
            help="Quick is fastest. Accurate uses beam search and takes a little longer.",
        )

        has_caption = bool(st.session_state.get("caption"))
        clicked = st.button(
            "Regenerate caption" if has_caption else "Generate caption",
            disabled=image is None,
            use_container_width=True,
        )

        if clicked and image is not None:
            try:
                with st.spinner("Describing your image…"):
                    start = time.time()
                    inputs = processor(images=image, return_tensors="pt").to(device)

                    gen_kwargs = {"max_new_tokens": 40}
                    if mode == "Accurate":
                        gen_kwargs.update(num_beams=5, early_stopping=True)

                    with torch.inference_mode():
                        output = model.generate(**inputs, **gen_kwargs)

                    caption = processor.decode(output[0], skip_special_tokens=True).strip()
                    st.session_state["caption"] = caption
                    st.session_state["meta"] = {
                        "seconds": time.time() - start,
                        "mode": mode,
                        "words": len(caption.split()),
                    }
            except Exception as e:
                st.error(f"Caption generation failed: {e}")

        caption = st.session_state.get("caption")
        meta = st.session_state.get("meta") or {}

        if caption and image is not None:
            st.markdown(
                f"""
                <div class="result">
                    <div class="result-label">Generated caption</div>
                    <div class="result-text">{html.escape(caption)}</div>
                    <div class="meta">
                        <span class="chip">{meta.get('words', 0)} words</span>
                        <span class="chip">{meta.get('seconds', 0):.1f}s</span>
                        <span class="chip">{html.escape(str(meta.get('mode', '')))} mode</span>
                        <span class="chip">BLIP · Flickr8k</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.code(caption, language=None)
        elif image is not None:
            st.markdown(
                """
                <div class="state">
                    <div class="state-icon">◐</div>
                    <div class="state-title">Ready when you are</div>
                    <div class="state-text">Press Generate caption to describe this image.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="state">
                    <div class="state-icon">◐</div>
                    <div class="state-title">Upload an image to begin</div>
                    <div class="state-text">Add a photo on the left,<br>then generate its caption here.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
