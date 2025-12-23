# =========================================
# Animal Image Classification - Streamlit App
# FINAL VERSION (Open + Close Camera)
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
# Load Model & Dataset
# ---------------------------------
MODEL_PATH = "animal_classifier_model.h5"
DATASET_DIR = "animal_images"

model = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = sorted(os.listdir(DATASET_DIR))

# ---------------------------------
# Sidebar Navigation (Vertical)
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
        - Supports **15 animal classes**  
        - Works on **CPU-only systems**  
        - Supports image upload & live camera capture
        """
    )

    st.success("👉 Use the **Classify** tab from the left to start.")

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
    # Camera Control (Open / Close)
    # -----------------------------
    if "camera_open" not in st.session_state:
        st.session_state.camera_open = False

    col_open, col_close = st.columns(2)

    with col_open:
        if st.button("📸 Open Camera"):
            st.session_state.camera_open = True

    with col_close:
        if st.button("❌ Close Camera"):
            st.session_state.camera_open = False

    camera_image = None
    if st.session_state.camera_open:
        camera_image = st.camera_input("Take a photo")

    st.divider()

    # -----------------------------
    # Submit & Classify
    # -----------------------------
    if st.button("🚀 Submit & Classify"):

        images_to_predict = []

        # Uploaded images
        if uploaded_files:
            for file in uploaded_files:
                img = Image.open(file).convert("RGB")
                images_to_predict.append((file.name, img))

        # Camera image
        if camera_image is not None:
            img = Image.open(camera_image).convert("RGB")
            images_to_predict.append(("Camera Image", img))

        if not images_to_predict:
            st.warning("⚠️ Please upload an image or open the camera.")
        else:
            for name, image in images_to_predict:

                st.subheader(f"🖼️ {name}")
                st.image(image, width="stretch")

                # Preprocess
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # Predict
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
# ANALYSIS PAGE
# =====================================================
elif page == "📊 Analysis":

    st.title("📊 Model Performance Analysis")

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

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
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

    # -----------------------------
    # Accuracy & Loss Graphs
    # -----------------------------
    st.subheader("📈 Training Accuracy & Loss")

    epochs = list(range(1, 11))
    train_acc = [0.73, 0.92, 0.95, 0.98, 0.99, 0.99, 1.0, 1.0, 1.0, 1.0]
    val_acc = [0.85, 0.84, 0.83, 0.90, 0.91, 0.90, 0.90, 0.91, 0.91, 0.90]
    train_loss = [0.96, 0.23, 0.15, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005]
    val_loss = [0.48, 0.47, 0.51, 0.38, 0.35, 0.38, 0.36, 0.36, 0.34, 0.35]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_acc, label="Train Accuracy")
    ax1.plot(epochs, val_acc, label="Validation Accuracy")
    ax1.set_title("Accuracy")
    ax1.legend()

    ax2.plot(epochs, train_loss, label="Train Loss")
    ax2.plot(epochs, val_loss, label="Validation Loss")
    ax2.set_title("Loss")
    ax2.legend()

    st.pyplot(fig)

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
