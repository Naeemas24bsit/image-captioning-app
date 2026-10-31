import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide"
)

MODEL_ID = "naeemajalal/image-captioning-blip-flickr8k"

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model

st.title("AI Image Caption Generator")
st.write("Upload an image and let Artificial Intelligence describe it.")

try:
    with st.spinner("Loading AI model..."):
        processor, model = load_model()

    st.success("AI Model is ready!")

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Your Uploaded Image",
                 use_container_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Analyzing image..."):
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                )

                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=40
                    )

                caption = processor.decode(
                    output[0],
                    skip_special_tokens=True
                )

            st.subheader("Generated Caption")
            st.success(caption)

except Exception as e:
    st.error(f"Unable to load model: {e}")
