%%writefile app.py
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(
    page_title="Medical Eye Disease Detection System",
    page_icon="👁️",
    layout="wide"
)

CLASS_NAMES = [
    "Cataract",
    "Jaundice",
    "Normal_Eye",
    "Pterygium"
]

@st.cache_resource
def load_ai_model():
    return load_model(
        "/content/drive/MyDrive/Klasifikasi Mata/final_model.keras"
    )

model = load_ai_model()

st.sidebar.title("🏥 Medical Eye AI System")

st.title("👁️ Medical Eye Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload Eye Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image")

    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    probs = prediction[0]

    pred_idx = np.argmax(probs)

    pred_class = CLASS_NAMES[pred_idx]

    confidence = probs[pred_idx] * 100

    with col2:

        st.subheader("Detection Result")

        st.success(
            f"{pred_class} ({confidence:.2f}%)"
        )

    st.subheader("Probability")

    for i, disease in enumerate(CLASS_NAMES):

        st.progress(float(probs[i]))

        st.write(
            f"{disease}: {probs[i]*100:.2f}%"
        )

    fig, ax = plt.subplots()

    ax.bar(
        CLASS_NAMES,
        probs * 100
    )

    ax.set_ylabel(
        "Probability (%)"
    )

    st.pyplot(fig)

    st.subheader("ROI Segmentation")

    img_cv = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        img_cv,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY
    )

    st.image(
        thresh,
        caption="Segmented ROI"
    )
