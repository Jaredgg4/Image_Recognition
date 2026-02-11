import streamlit as st
from PIL import Image
from model_classifier import ImageClassifier

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Classification App")
st.write("Upload an image to classify it using deep learning")

st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["resnet50", "resnet101", "vit_b_16"]
)

top_k = st.sidebar.slider("Number of predictions", 1, 10, 5)

@st.cache_resource
def load_classifier(model_name):
    return ImageClassifier(model_name)

classifier = load_classifier(model_choice)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Predictions")
        with st.spinner("Classifying..."):
            predictions = classifier.predict(image, top_k)
        
        for i, (class_name, prob) in enumerate(predictions, 1):
            st.write(f"**{i}. {class_name}**")
            st.progress(prob)
            st.write(f"Confidence: {prob*100:.2f}%")
            st.write("---")

with st.expander("ℹ️ How to use"):
    st.write("""
    1. Select a model from the sidebar
    2. Upload an image (JPG, JPEG, or PNG)
    3. View the classification results
    """)