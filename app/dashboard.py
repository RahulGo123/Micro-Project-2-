import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Satellite Mission Control", layout="centered")


@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("models/satellite_resnet_finetuned.keras")
    except:
        model = tf.keras.models.load_model("models/satellite_resnet.keras")
    return model


with st.spinner("Booting up AI Systems..."):
    model = load_model()

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

st.title("🛰️ Satellite Imagery Analysis")
st.markdown("Upload a Sentinel-2 satellite image to classify the terrain type.")

with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)

    img_batch = np.expand_dims(img_array, axis=0)

    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_batch)

    predictions = model.predict(img_preprocessed)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions)

    with col2:
        st.subheader("AI analysis")

        if confidence > 0.90:
            st.success(f"**Class:** {predicted_class}")
        elif confidence > 0.60:
            st.warning(f"**Class:** {predicted_class}")
        else:
            st.error(f"**Class:** {predicted_class} (Unsure)")

        st.metric("Confidence Score", f"{confidence:.2%}")

        st.markdown("### Detailed Probability")
        st.bar_chart(data=predictions[0])
        # (Optional: Map index to name in chart if desired, but this is quick)
        with st.expander("See Class Key"):
            st.write(dict(enumerate(CLASS_NAMES)))

else:
    st.info("👈 Please upload an image from the sidebar to begin.")
