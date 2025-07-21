import base64
import io
import os
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import requests
import streamlit as st
import torch

# Import configuration
from config import (
    APP_CONTENT,
    CUSTOM_CSS,
    DOWNLOAD_CONFIG,
    EXAMPLE_IMAGES,
    MODEL_CONFIG,
    OVERLAY_CONFIG,
    PAGE_CONFIG,
    UPLOAD_CONFIG,
)
from PIL import Image

# Import local model files
from unet_model import UNet

# Try to import preprocessing from data_setup, but provide fallback if rasterio is not available
try:
    from data_setup import get_preprocessing
except ImportError as e:
    st.warning(f"Could not import from data_setup: {e}")
    st.info("Using fallback preprocessing function (rasterio not available)")

    def get_preprocessing():
        """Fallback preprocessing function when rasterio is not available"""

        def to_float_and_normalize(image, **kwargs):
            return image.astype(np.float32) / 255.0

        def transpose_to_chw(image, **kwargs):
            return image.transpose(2, 0, 1)

        def to_float32(mask, **kwargs):
            return mask.astype(np.float32)

        _transform = [
            # Scale to [0,1] range
            A.Lambda(image=to_float_and_normalize),
            # Convert to PyTorch format (CHW)
            A.Lambda(image=transpose_to_chw),
            # Ensure mask is float32
            A.Lambda(mask=to_float32),
        ]
        return A.Compose(_transform)


# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def download_model_from_gdrive(file_id, destination):
    """Download model file from Google Drive with improved handling"""
    import re

    try:
        st.info(
            "🔄 Downloading model file from Google Drive... This may take a few minutes."
        )

        progress_bar = st.progress(0)
        status_text = st.empty()
        session = requests.Session()

        # Method 1: Try direct download
        status_text.text("Attempting direct download...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = session.get(url, stream=False)

        # Check if we got a confirmation page
        if (
            "text/html" in response.headers.get("content-type", "")
            and "virus scan" in response.text.lower()
        ):
            # Extract form parameters from virus scan page
            status_text.text("Handling virus scan confirmation...")

            # Extract the form action URL and parameters
            action_match = re.search(r'action="([^"]*)"', response.text)
            id_match = re.search(r'name="id" value="([^"]*)"', response.text)
            export_match = re.search(r'name="export" value="([^"]*)"', response.text)
            confirm_match = re.search(r'name="confirm" value="([^"]*)"', response.text)
            uuid_match = re.search(r'name="uuid" value="([^"]*)"', response.text)

            if action_match and id_match and confirm_match:
                action_url = action_match.group(1)
                file_id_param = id_match.group(1)
                export_param = export_match.group(1) if export_match else "download"
                confirm_param = confirm_match.group(1)
                uuid_param = uuid_match.group(1) if uuid_match else ""

                # Build download URL with all parameters
                download_url = f"{action_url}?id={file_id_param}&export={export_param}&confirm={confirm_param}"
                if uuid_param:
                    download_url += f"&uuid={uuid_param}"

                status_text.text("Downloading with confirmation parameters...")
                response = session.get(download_url, stream=True)
            else:
                raise Exception(
                    "Could not extract form parameters from virus scan page"
                )
        else:
            # Direct download worked, get streaming response
            response = session.get(url, stream=True)

        # Verify we got binary content
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            # Still getting HTML, try alternative approach
            status_text.text("Trying alternative download method...")
            alt_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
            response = session.get(alt_url, stream=True)

            # If still HTML, fail
            if "text/html" in response.headers.get("content-type", ""):
                raise Exception("Cannot bypass Google Drive download restrictions")

        # Download the file
        file_size = response.headers.get("content-length")
        if file_size:
            file_size = int(file_size)

        downloaded = 0
        chunk_size = DOWNLOAD_CONFIG["chunk_size"]

        status_text.text("Downloading model file...")

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Update progress
                    if file_size and file_size > 0:
                        progress = min(downloaded / file_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(
                            f"Downloaded: {downloaded / (1024*1024):.1f} MB / {file_size / (1024*1024):.1f} MB"
                        )
                    else:
                        status_text.text(
                            f"Downloaded: {downloaded / (1024*1024):.1f} MB"
                        )

        # Verify download
        if (
            downloaded < DOWNLOAD_CONFIG["min_file_size"]
        ):  # Less than configured minimum - probably not the model
            os.remove(destination)
            raise Exception(
                f"Downloaded file too small ({downloaded} bytes) - likely not the model file"
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Download completed!")
        st.success(f"Model downloaded successfully ({downloaded / (1024*1024):.1f} MB)")
        return True

    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        st.info(
            "You may need to manually download the model file and place it in the project directory."
        )
        return False


@st.cache_resource
def load_model():
    """Load the trained UNet model (cached for performance)"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model = UNet(
            in_channels=MODEL_CONFIG["input_channels"],
            out_channels=MODEL_CONFIG["output_channels"],
        )

        # Model file path
        model_path = MODEL_CONFIG["model_path"]

        # Check if model file exists, if not download it
        if not os.path.exists(model_path):
            st.warning(
                f"Model file not found at {model_path}. Attempting to download from Google Drive..."
            )

            # Google Drive file ID from the provided URL
            file_id = MODEL_CONFIG["google_drive_file_id"]

            # Download the model
            if not download_model_from_gdrive(file_id, model_path):
                st.error("Failed to download model file.")
                return None, None

        # Load trained weights
        if os.path.exists(model_path):
            st.info("📦 Loading model weights...")
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=False)
            )
            model.to(device)
            model.eval()
            st.success("✅ Model loaded successfully!")
            return model, device
        else:
            st.error(f"Model file still not found at {model_path}")
            return None, None

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


@st.cache_data
def process_example_images():
    """Process example images once and cache the results"""
    example_files = EXAMPLE_IMAGES["files"]
    example_captions = EXAMPLE_IMAGES["captions"]

    # Process example images (without model dependency for caching)
    example_images = []

    for file in example_files:
        try:
            img = Image.open(file)
            example_images.append(img)
        except Exception as e:
            example_images.append(None)

    return example_images, example_captions


@st.cache_data
def process_example_overlays(_model, device):
    """Process example overlays with model (cached separately)"""
    example_files = EXAMPLE_IMAGES["files"]
    example_overlays = []

    for file in example_files:
        try:
            img = Image.open(file)
            # Process the image
            processed_image, resized_image = preprocess_image(img)
            if processed_image is not None:
                prediction = run_inference(_model, device, processed_image)
                if prediction is not None:
                    overlay = create_overlay(resized_image, prediction)
                    example_overlays.append(overlay)
                else:
                    example_overlays.append(None)
            else:
                example_overlays.append(None)
        except Exception as e:
            example_overlays.append(None)

    return example_overlays


def preprocess_image(image):
    """Preprocess uploaded image for model inference"""
    try:
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(MODEL_CONFIG["input_size"], Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.array(image)

        # Apply preprocessing (same as training)
        preprocessing = get_preprocessing()
        if preprocessing:
            sample = preprocessing(
                image=image_array, mask=np.zeros(MODEL_CONFIG["input_size"])
            )
            processed_image = sample["image"]  # Shape: (3, 512, 512)
        else:
            # Fallback preprocessing
            processed_image = image_array.astype(np.float32) / 255.0
            processed_image = processed_image.transpose(2, 0, 1)

        return processed_image, image

    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None


def run_inference(model, device, processed_image):
    """Run model inference on preprocessed image"""
    try:
        # Convert to tensor
        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output) > MODEL_CONFIG["prediction_threshold"]
            prediction = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims

        return prediction

    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return None


def create_overlay(original_image, prediction):
    """Create overlay of original image with prediction mask"""
    try:
        # Convert prediction to PIL image
        prediction_image = (prediction * 255).astype(np.uint8)
        prediction_pil = Image.fromarray(prediction_image)

        # Create overlay
        overlay = original_image.copy()
        prediction_rgba = prediction_pil.convert("RGBA")

        # Create red overlay for buildings
        red_overlay = Image.new(
            "RGBA", prediction_rgba.size, OVERLAY_CONFIG["building_color"]
        )

        # Apply red overlay where prediction is white
        prediction_array = np.array(prediction_rgba)
        red_array = np.array(red_overlay)

        # Where prediction is white (255), apply red overlay
        mask = prediction_array[:, :, 0] > OVERLAY_CONFIG["mask_threshold"]
        overlay_array = np.array(overlay.convert("RGBA"))
        overlay_array[mask] = red_array[mask]

        return Image.fromarray(overlay_array)

    except Exception as e:
        st.error(f"Error creating overlay: {str(e)}")
        return None


def main():
    # Header
    st.markdown(
        f'<h1 class="main-header">{APP_CONTENT["main_title"]}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sub-header">{APP_CONTENT["subtitle"]}</p>',
        unsafe_allow_html=True,
    )

    # Info box
    st.markdown(APP_CONTENT["info_box_content"], unsafe_allow_html=True)

    # Load model (cached as resource)
    model, device = load_model()

    # Process example images (cached)
    with st.spinner("Loading AI model and processing examples..."):
        example_images, example_captions = process_example_images()

        # Process overlays if model is available
        if model is not None:
            example_overlays = process_example_overlays(model, device)
        else:
            example_overlays = [None, None, None]

    # Example images section
    st.markdown("**Example drone images (what to expect):**")
    cols = st.columns(3)
    for i, (img, caption) in enumerate(zip(example_images, example_captions)):
        if img is not None:
            cols[i].image(img, caption=caption, use_container_width=True)
        else:
            cols[i].write(f"Could not load example {i+1}")

    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        st.warning("Example overlays cannot be shown without a loaded model.")
    else:
        # Show overlays for example images in a second row
        st.markdown("**Model predictions for the example images:**")
        overlay_cols = st.columns(3)
        for i, (overlay, caption) in enumerate(zip(example_overlays, example_captions)):
            if overlay is not None:
                overlay_cols[i].image(
                    overlay,
                    caption=f"Overlay for {caption}",
                    use_container_width=True,
                )
            else:
                overlay_cols[i].write(f"Could not process {caption}.")

    # File upload
    st.markdown("## 📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=UPLOAD_CONFIG["allowed_types"],
        help=UPLOAD_CONFIG["help_text"],
    )

    if uploaded_file is not None:
        # Display original image
        st.markdown("## 📸 Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_container_width=True)

        # Process button
        if st.button("🚀 Process Image", type="primary"):
            with st.spinner("Processing image..."):
                # Get device from model
                device = next(model.parameters()).device

                # Preprocess image
                processed_image, resized_image = preprocess_image(original_image)

                if processed_image is not None:
                    # Run inference
                    prediction = run_inference(model, device, processed_image)

                    if prediction is not None:
                        # Create overlay
                        overlay = create_overlay(resized_image, prediction)

                        # Display results
                        st.markdown("## 🎯 Results")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("### Original (Resized)")
                            st.image(resized_image, use_container_width=True)

                        with col2:
                            st.markdown("### Building Mask")
                            prediction_display = (prediction * 255).astype(np.uint8)
                            st.image(
                                prediction_display,
                                use_container_width=True,
                                caption="White = Buildings, Black = Background",
                            )

                        with col3:
                            st.markdown("### Overlay View")
                            if overlay:
                                st.image(
                                    overlay,
                                    use_container_width=True,
                                    caption="Red overlay shows detected buildings",
                                )

                        # Download buttons
                        st.markdown("## 💾 Download Results")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Download original
                            buf = io.BytesIO()
                            resized_image.save(buf, format="PNG")
                            st.download_button(
                                label="Download Original",
                                data=buf.getvalue(),
                                file_name=f"original_{uploaded_file.name}",
                                mime="image/png",
                            )

                        with col2:
                            # Download prediction
                            buf = io.BytesIO()
                            prediction_pil = Image.fromarray(prediction_display)
                            prediction_pil.save(buf, format="PNG")
                            st.download_button(
                                label="Download Mask",
                                data=buf.getvalue(),
                                file_name=f"prediction_{uploaded_file.name}",
                                mime="image/png",
                            )

                        with col3:
                            # Download overlay
                            if overlay:
                                buf = io.BytesIO()
                                overlay.save(buf, format="PNG")
                                st.download_button(
                                    label="Download Overlay",
                                    data=buf.getvalue(),
                                    file_name=f"overlay_{uploaded_file.name}",
                                    mime="image/png",
                                )

                        # Update session state
                        st.session_state.processed_images += 1

                        # Success message
                        st.success("✅ Image processed successfully!")

                        # Model info
                        with st.expander("ℹ️ Model Information"):
                            st.write(f"**Model:** U-Net Architecture")
                            st.write(
                                f"**Input Size:** {MODEL_CONFIG['input_size'][0]}x{MODEL_CONFIG['input_size'][1]} pixels"
                            )
                            st.write(f"**Device:** {device}")
                            st.write(f"**Task:** Binary building segmentation")

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        st.markdown("Use the pages in the sidebar to explore the application:")

        st.markdown("### 📄 Available Pages")
        st.markdown(
            """
        - **Main App** - Upload and process images
        - **About** - Model details and performance metrics
        """
        )

        st.markdown("---")

        st.markdown("### 🔧 Quick Actions")
        if st.button("🔄 Refresh App"):
            st.rerun()

        st.markdown("### ℹ️ Tips")
        st.info(
            """
        💡 **Pro Tips:**
        - Upload high-resolution images for better results
        - Try the example images first to understand the output
        - Check the About page for technical details
        """
        )

        st.markdown("---")
        st.markdown("### 📊 Current Session")

        # Session state info
        if "processed_images" not in st.session_state:
            st.session_state.processed_images = 0

        st.metric(label="Images Processed", value=st.session_state.processed_images)


if __name__ == "__main__":
    main()
