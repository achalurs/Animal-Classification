# =========================================
# Animal Image Classification - Streamlit App
# Cloud-Safe Final Version
# =========================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="Animal Classification", layout="wide")

# ---------------------------------
# Load Model
# ---------------------------------
MODEL_PATH = "animal_classifier_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------
# Hardcoded Class Names (Cloud Safe)
# ---------------------------------
CLASS_NAMES = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe",
    "Horse", "Kangaroo", "Lion", "Panda",
    "Tiger", "Zebra"
]

# Dataset folder (used only locally)
DATASET_DIR = "animal_images"

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📸 Classify", "📊 Analysis"]
)

# =====================================================
# HOME PAGE
# =====================================================
if page == "🏠 Home":

    st.title("🐾 Animal Image Classification System")

    st.markdown(
        """
        ### ⚠️ Only the following animals will be classified:
        **Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant,  
        Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra**
        """
    )

    st.markdown(
        """
        ### 📌 Project Overview
        - Internship-level Deep Learning project  
        - Transfer Learning using **MobileNetV2**  
        - 15 Animal Classes  
        - CPU-only optimized  
        - Image upload & live camera prediction
        """
    )

    st.success("👉 Use the **Classify** tab from the left to start prediction.")

# =====================================================
# CLASSIFY PAGE
# =====================================================
elif page == "📸 Classify":

    st.title("📸 Animal Classification")

    # -----------------------------
    # Upload Images
    # -----------------------------
    st.subheader("📂 Upload Images")
    uploaded_files = st.file_uploader(
        "Upload one or multiple images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )

    st.divider()

    # -----------------------------
    # Camera Controls
    # -----------------------------
    if "camera_open" not in st.session_state:
        st.session_state.camera_open = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Open Camera"):
            st.session_state.camera_open = True

    with col2:
        if st.button("❌ Close Camera"):
            st.session_state.camera_open = False

    camera_image = None
    if st.session_state.camera_open:
        camera_image = st.camera_input("Take a photo")

    st.divider()

    # -----------------------------
    # Submit & Predict
    # -----------------------------
    if st.button("🚀 Submit & Classify"):

        images_to_predict = []

        if uploaded_files:
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                images_to_predict.append((file.name, img))

        if camera_image is not None:
            img = Image.open(camera_image).convert("RGB")
            images_to_predict.append(("Camera Image", img))

        if not images_to_predict:
            st.warning("⚠️ Please upload an image or use the camera.")
        else:
            for name, image in images_to_predict:

                st.subheader(f"🖼️ {name}")
                st.image(image, width="stretch")

                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = model.predict(img_array)[0]
                top_3_idx = preds.argsort()[-3:][::-1]

                st.markdown("### 🥇 Top-3 Predictions")
                for i, idx in enumerate(top_3_idx, start=1):
                    st.write(
                        f"**{i}. {CLASS_NAMES[idx]}** — Confidence: `{preds[idx]*100:.2f}%`"
                    )

                st.success(f"✅ Final Prediction: **{CLASS_NAMES[top_3_idx[0]]}**")
                st.divider()

# =====================================================
# ANALYSIS PAGE (LOCAL ONLY)
# =====================================================
elif page == "📊 Analysis":

    st.title("📊 Model Performance Analysis")

    if not os.path.exists(DATASET_DIR):
        st.warning("📁 Dataset not available in cloud deployment.")
        st.info("Confusion matrix and training graphs are available in local execution.")
    else:
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        val_data = datagen.flow_from_directory(
            DATASET_DIR,
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )

        y_true = val_data.classes
        y_pred_probs = model.predict(val_data)
        y_pred = np.argmax(y_pred_probs, axis=1)

        st.subheader("🔲 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax_cm
        )

        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_title("Confusion Matrix – Animal Classification")

        st.pyplot(fig_cm)

# ---------------------------------
# Footer
# ---------------------------------
st.divider()
st.markdown(
    """
    **Internship Project – Animal Image Classification**  
    *MobileNetV2 | TensorFlow | Streamlit*
    """
)
