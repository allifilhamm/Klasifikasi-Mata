import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gdown
import os
import tensorflow as tf

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
    model_path = "final_model.keras"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1SrpLuT4TGg7K7_qYHD7qHu6gdtJncPwI"
        gdown.download(url, model_path, quiet=False)

    return load_model(model_path, compile=False)

model = load_ai_model()

def generate_gradcam(img_array, model, sub_model_name="resnet50", last_conv_layer_name="conv5_block3_out"):
    resnet_model = model.get_layer(sub_model_name)
    grad_model = tf.keras.models.Model(
        inputs=[resnet_model.inputs],
        outputs=[resnet_model.get_layer(last_conv_layer_name).output, resnet_model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, resnet_features = grad_model(img_array)
        x = model.get_layer("global_average_pooling2d")(resnet_features)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x)
        preds = model.get_layer("dense_1")(x)
        
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_gradcam(img_pil, heatmap, alpha=0.45):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img_cv
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

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
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)
    probs = prediction[0]
    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    with col2:
        st.subheader("Detection Result")
        st.success(f"{pred_class} ({confidence:.2f}%)")

    st.subheader("Probability")
    for i, disease in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]))
        st.write(f"{disease}: {probs[i]*100:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, probs * 100)
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)

    st.markdown("---")
    
    col_seg, col_cam = st.columns(2)

    with col_seg:
        st.subheader("ROI Segmentation (Otsu Method)")
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        st.image(
            thresh,
            caption="Segmented ROI (Sklera & Iris Tetap Terjaga)",
            use_container_width=True
        )

    with col_cam:
        st.subheader("Grad-CAM Visualization")
        try:
            heatmap = generate_gradcam(img_array, model, sub_model_name="resnet50", last_conv_layer_name="conv5_block3_out")
            gradcam_result = superimpose_gradcam(image, heatmap, alpha=0.45)
            st.image(
                gradcam_result,
                caption=f"Grad-CAM Heatmap untuk Deteksi: {pred_class}",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Gagal memuat visualisasi Grad-CAM: {e}")
