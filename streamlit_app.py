import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import base64
import sys
import os

# Import local model files
from unet_model import UNet
from data_setup import get_preprocessing

# Page configuration
st.set_page_config(
    page_title="Land Cover Classification",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #2c3e50;
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 2rem;
        color: white;
    }
    
    .info-box h3 {
        color: #ecf0f1;
        margin-bottom: 10px;
    }
    
    .info-box ul {
        list-style: none;
        padding-left: 0;
    }
    
    .info-box li {
        padding: 5px 0;
        color: #bdc3c7;
    }
    
    .info-box li:before {
        content: "✓ ";
        color: #27ae60;
        font-weight: bold;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained UNet model (cached for performance)"""
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        model = UNet(in_channels=3, out_channels=1)
        
        # Load trained weights
        model_path = 'best_unet_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, device
        else:
            st.error(f"Model file not found at {model_path}")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image):
    """Preprocess uploaded image for model inference"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (512x512)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply preprocessing (same as training)
        preprocessing = get_preprocessing()
        if preprocessing:
            sample = preprocessing(image=image_array, mask=np.zeros((512, 512)))
            processed_image = sample['image']  # Shape: (3, 512, 512)
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
            prediction = torch.sigmoid(output) > 0.5
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
        prediction_rgba = prediction_pil.convert('RGBA')
        
        # Create red overlay for buildings
        red_overlay = Image.new('RGBA', prediction_rgba.size, (255, 0, 0, 128))
        
        # Apply red overlay where prediction is white
        prediction_array = np.array(prediction_rgba)
        red_array = np.array(red_overlay)
        
        # Where prediction is white (255), apply red overlay
        mask = prediction_array[:, :, 0] > 127
        overlay_array = np.array(overlay.convert('RGBA'))
        overlay_array[mask] = red_array[mask]
        
        return Image.fromarray(overlay_array)
        
    except Exception as e:
        st.error(f"Error creating overlay: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 Land Cover Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered building detection from drone imagery</p>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        <h3>How it works:</h3>
        <ul>
            <li>Upload a drone image (JPG, PNG, or TIF format)</li>
            <li>Our AI model will analyze the image and detect buildings</li>
            <li>View the original image, prediction mask, and overlay</li>
            <li>Download the results for further analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Example images section
    st.markdown("**Example drone images (what to expect):**")
    example_files = ["example_1.tif", "example_2.tif", "example_3.tif"]
    example_captions = ["Example 1", "Example 2", "Example 3"]
    example_images = []
    cols = st.columns(3)
    for i, (file, caption) in enumerate(zip(example_files, example_captions)):
        try:
            img = Image.open(file)
            example_images.append(img)
            cols[i].image(img, caption=caption, use_container_width=True)
        except Exception as e:
            example_images.append(None)
            cols[i].write(f"Could not load {file}")

    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()

    if model is None:
        st.error("Failed to load model. Please check the model file path.")
        st.warning("Example overlays cannot be shown without a loaded model.")
    else:
        # Show overlays for example images in a second row
        st.markdown("**Model predictions for the example images:**")
        overlay_cols = st.columns(3)
        for i, img in enumerate(example_images):
            if img is not None:
                processed_image, resized_image = preprocess_image(img)
                if processed_image is not None:
                    prediction = run_inference(model, device, processed_image)
                    if prediction is not None:
                        overlay = create_overlay(resized_image, prediction)
                        if overlay:
                            overlay_cols[i].image(overlay, caption=f"Overlay for {example_captions[i]}", use_container_width=True)
                        else:
                            overlay_cols[i].write("Could not create overlay.")
                    else:
                        overlay_cols[i].write("Prediction failed.")
                else:
                    overlay_cols[i].write("Preprocessing failed.")
            else:
                overlay_cols[i].write("No image loaded.")
    
    # File upload
    st.markdown("## 📁 Upload Your Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="Upload a drone image to detect buildings"
    )
    
    if uploaded_file is not None:
        # Display original image
        st.markdown("## 📸 Original Image")
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_container_width=True)
        
        # Process button
        if st.button("🚀 Process Image", type="primary"):
            with st.spinner("Processing image..."):
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
                            st.image(prediction_display, use_container_width=True, caption="White = Buildings, Black = Background")
                        
                        with col3:
                            st.markdown("### Overlay View")
                            if overlay:
                                st.image(overlay, use_container_width=True, caption="Red overlay shows detected buildings")
                        
                        # Download buttons
                        st.markdown("## 💾 Download Results")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Download original
                            buf = io.BytesIO()
                            resized_image.save(buf, format='PNG')
                            st.download_button(
                                label="Download Original",
                                data=buf.getvalue(),
                                file_name=f"original_{uploaded_file.name}",
                                mime="image/png"
                            )
                        
                        with col2:
                            # Download prediction
                            buf = io.BytesIO()
                            prediction_pil = Image.fromarray(prediction_display)
                            prediction_pil.save(buf, format='PNG')
                            st.download_button(
                                label="Download Mask",
                                data=buf.getvalue(),
                                file_name=f"prediction_{uploaded_file.name}",
                                mime="image/png"
                            )
                        
                        with col3:
                            # Download overlay
                            if overlay:
                                buf = io.BytesIO()
                                overlay.save(buf, format='PNG')
                                st.download_button(
                                    label="Download Overlay",
                                    data=buf.getvalue(),
                                    file_name=f"overlay_{uploaded_file.name}",
                                    mime="image/png"
                                )
                        
                        # Success message
                        st.success("✅ Image processed successfully!")
                        
                        # Model info
                        with st.expander("ℹ️ Model Information"):
                            st.write(f"**Model:** U-Net Architecture")
                            st.write(f"**Input Size:** 512x512 pixels")
                            st.write(f"**Device:** {device}")
                            st.write(f"**Task:** Binary building segmentation")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ About")
        st.markdown("""
        This application uses a trained U-Net model to detect buildings in drone imagery.
        
        **Features:**
        - Real-time building detection
        - High-resolution mask generation
        - Interactive overlay visualization
        - Download results
        
        **Technical Details:**
        - Model: U-Net with skip connections
        - Input: RGB images (512x512)
        - Output: Binary building masks
        - Framework: PyTorch
        """)
        
        st.markdown("## 📊 Model Performance")
        st.markdown("""
        - **Architecture:** U-Net
        - **Training Data:** Drone imagery patches
        - **Validation Dice Score:** ~0.81
        - **Validation IoU Score:** ~0.68
        """)
        
        st.markdown("## 🔗 Links")
        st.markdown("""
        - [GitHub Repository](#)
        - [Model Training Notebook](#)
        - [Documentation](#)
        """)

if __name__ == "__main__":
    main() 