
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd

from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="Warehouse Product Sorting",
    page_icon="📦",
    layout="wide"
)

# Load trained model

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "warehouse_cnn.keras"
    )

model = load_model()

# Load category names

with open("class_names.json", "r") as f:

    class_names = json.load(f)

# Sorting bins

sorting_bins = {

    "Apple": "BIN 1 - Apples",

    "Cherry": "BIN 2 - Cherries",

    "Tomato": "BIN 3 - Tomatoes"

}

st.title("📦 AI Warehouse Product Sorting System")

st.write(
    "Automated product classification using CNN"
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Product",
        width=300
    )

    # Preprocess image

    image = image.resize((100, 100))

    img_array = np.array(image)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict product

    predictions = model.predict(
        img_array,
        verbose=0
    )

    index = np.argmax(predictions[0])

    product = class_names[index]

    confidence = float(
        predictions[0][index]
    )

    # Sorting decision

    if confidence < 0.60:

        destination = "MANUAL REVIEW"

    else:

        destination = sorting_bins.get(
            product,
            "MANUAL REVIEW"
        )

    st.divider()

    st.subheader("Classification Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Product",
        product
    )

    col2.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    col3.metric(
        "Sorting Destination",
        destination
    )

    st.success(
        f"Sorting decision: {destination}"
    )

    # Store sorting history

    if "history" not in st.session_state:

        st.session_state.history = []

    if st.button("Add to Sorting Report"):

        st.session_state.history.append({

            "Product": product,

            "Confidence": round(
                confidence * 100, 2
            ),

            "Destination": destination,

            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        })

        st.success("Added to sorting report!")

# Display history

st.divider()

st.subheader("Warehouse Sorting History")

if "history" in st.session_state and st.session_state.history:

    df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Sorting Report",
        csv,
        "warehouse_sorting_report.csv",
        "text/csv"
    )

else:

    st.info("No sorting records yet.")
