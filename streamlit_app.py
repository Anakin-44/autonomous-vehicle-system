import streamlit as st
from PIL import Image, UnidentifiedImageError
import app as backend

st.set_page_config(
    page_title="Autonomous Vehicle Intelligence System",
    page_icon="🚘",
    layout="wide",
)

st.title("🚘 Autonomous Vehicle Intelligence System")
st.caption(
    "AI-powered object detection, lane analysis, "
    "traffic sign recognition, and driver behaviour classification."
)

@st.cache_resource(show_spinner="Loading the four trained models...")
def initialize_models():
    backend.load_models()
    return backend.models, backend.model_errors


st.subheader("Upload images")

col1, col2 = st.columns(2)

with col1:
    dashcam_file = st.file_uploader(
        "Road / dashcam image",
        type=["jpg", "jpeg", "png", "webp"],
        key="dashcam",
    )

with col2:
    cabin_file = st.file_uploader(
        "Driver / cabin image",
        type=["jpg", "jpeg", "png", "webp"],
        key="cabin",
    )

if st.button("Analyze Images", type="primary", use_container_width=True):
    if dashcam_file is None or cabin_file is None:
        st.warning("Please upload both images before analyzing.")
        st.stop()

    try:
        with Image.open(dashcam_file) as source:
            dashcam_image = source.convert("RGB")

        with Image.open(cabin_file) as source:
            cabin_image = source.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("One of the uploaded files is not a valid image.")
        st.stop()

    try:
        models, errors = initialize_models()
    except Exception as exc:
        st.error(f"Model initialization failed: {exc}")
        st.stop()

    st.subheader("Model status")

    status_cols = st.columns(4)
    model_names = {
        "yolo": "Object Detection",
        "behavior": "Driver Behaviour",
        "lane": "Lane Detection",
        "sign": "Traffic Sign",
    }

    for column, key in zip(status_cols, model_names):
        if models[key] is not None:
            column.success(f"{model_names[key]}: Loaded")
        else:
            column.error(f"{model_names[key]}: Unavailable")
            if key in errors:
                column.caption(str(errors[key])[:250])

    tasks = [
        ("Object Detection", backend.detect_objects, dashcam_image),
        ("Driver Behaviour", backend.detect_behavior, cabin_image),
        ("Traffic Sign Recognition", backend.detect_signs, dashcam_image),
        ("Lane Detection", backend.detect_lanes, dashcam_image),
    ]

    st.subheader("Analysis results")

    result_cols = st.columns(2)

    for index, (name, function, image) in enumerate(tasks):
        with result_cols[index % 2]:
            st.markdown(f"#### {name}")

            try:
                result = function(image)

                if result.get("status") == "success":
                    st.success(result.get("summary", "Analysis completed."))
                    st.json(result)
                else:
                    st.error(result.get("error", "Analysis failed."))
            except Exception as exc:
                st.error(f"Could not run {name}: {exc}")