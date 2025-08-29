"""
About page for the Land Cover Classification Streamlit App.

Contains detailed information about the model, performance metrics, and links.
"""

import streamlit as st
from config import APP_CONTENT, CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="About - Land Cover Classification",
    page_icon="ℹ️",
    layout="wide",
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Header
st.markdown(
    f'<h1 class="main-header">About the Project</h1>',
    unsafe_allow_html=True,
)

# Main content in columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # About section
    st.markdown("## About")
    st.markdown(APP_CONTENT["sidebar_about"])

    # Technical Details
    st.markdown("## Technical Details")
    st.markdown(
        """
    **Model Architecture:**
    - U-Net with encoder-decoder structure
    - Skip connections for feature preservation
    - Batch normalization and ReLU activations
    - Final sigmoid activation for binary classification
    
    **Training Process:**
    - Dataset: High-resolution drone imagery patches
    - Augmentation: Rotation, flipping, color jittering
    - Loss function: Dice loss + Binary Cross Entropy
    - Optimizer: Adam with learning rate scheduling
    - Training time: ~6 hours on GPU
    
    **Inference:**
    - Input preprocessing: Resize to 512x512, normalize to [0,1]
    - Model prediction: Binary segmentation mask
    - Post-processing: Threshold at 0.5, overlay generation
    """
    )

with col2:
    # Performance metrics
    st.markdown("## Model Performance")
    st.markdown(APP_CONTENT["sidebar_performance"])

    # Performance visualization
    st.markdown("### Metrics Breakdown")

    # Create metrics display
    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(label="Dice Score", value="0.81", delta="0.05 vs baseline")
        st.metric(label="IoU Score", value="0.68", delta="0.08 vs baseline")

    with metric_col2:
        st.metric(label="Precision", value="0.84", delta="0.03 vs baseline")
        st.metric(label="Recall", value="0.79", delta="0.06 vs baseline")

# Links section
st.markdown("## Links & Resources")

link_col1, link_col2, link_col3 = st.columns(3)

with link_col1:
    st.markdown(
        """
    **📚 Documentation**
    - [GitHub Repository](https://github.com/your-username/Drone-Image-DL-Classification)
    - [Model Training Notebook](#)
    - [Dataset Information](#)
    """
    )

with link_col2:
    st.markdown(
        """
    **🤖 Model Resources**
    - [U-Net Paper](https://arxiv.org/abs/1505.04597)
    - [PyTorch Documentation](https://pytorch.org/docs/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """
    )

with link_col3:
    st.markdown(
        """
    **📊 Related Work**
    - [Building Detection Survey](#)
    - [Remote Sensing Papers](#)
    - [Computer Vision Resources](#)
    """
    )

# Model Architecture Diagram
st.markdown("## Model Architecture")

st.markdown(
    """
The U-Net architecture consists of:

**Encoder (Downsampling):**
- 4 encoding blocks with convolution + pooling
- Feature maps: 64 → 128 → 256 → 512 → 1024 channels
- Spatial resolution decreases by factor of 2 each level

**Decoder (Upsampling):**
- 4 decoding blocks with upsampling + convolution  
- Skip connections from corresponding encoder levels
- Feature maps: 1024 → 512 → 256 → 128 → 64 channels
- Final 1x1 convolution for binary classification

**Key Features:**
- Skip connections preserve fine-grained spatial information
- Batch normalization for training stability
- ReLU activations throughout (except final sigmoid)
- ~31M trainable parameters
"""
)

# Usage Statistics (if you want to add this later)
st.markdown("## Usage Statistics")
st.info(
    "Usage analytics coming soon! This will show app usage patterns and popular features."
)

# Contact/Support section
st.markdown("## Support & Contact")

support_col1, support_col2 = st.columns(2)

with support_col1:
    st.markdown(
        """
    **🐛 Report Issues:**
    - [GitHub Issues](https://github.com/your-username/Drone-Image-DL-Classification/issues)
    - Email: your-email@example.com
    
    **💡 Feature Requests:**
    - Submit via GitHub Issues
    - Tag with "enhancement" label
    """
    )

with support_col2:
    st.markdown(
        """
    **📧 Contact:**
    - Developer: Your Name
    - Institution: Your University/Company
    - Research Group: Computer Vision Lab
    
    **🤝 Contributions:**
    - Fork the repository
    - Submit pull requests
    - Follow contribution guidelines
    """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        🏠 Land Cover Classification Project | Built with ❤️ using Streamlit & PyTorch
    </div>
    """,
    unsafe_allow_html=True,
)
