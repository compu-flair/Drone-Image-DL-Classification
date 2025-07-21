"""
Configuration file for the Land Cover Classification Streamlit App.

This module contains all configurable settings and constants used throughout the application.
"""

# Streamlit Page Configuration
PAGE_CONFIG = {
    "page_title": "Land Cover Classification",
    "page_icon": "🏠",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Model Configuration
MODEL_CONFIG = {
    "model_path": "best_unet_model.pth",
    "google_drive_file_id": "17mrNvHi3hXEDc4jE9yaqR4cTw1ek97MW",
    "input_channels": 3,
    "output_channels": 1,
    "input_size": (512, 512),
    "prediction_threshold": 0.5,
}

# Example Images Configuration
EXAMPLE_IMAGES = {
    "files": ["../img/example_1.tif", "../img/example_2.tif", "../img/example_3.tif"],
    "captions": ["Example 1", "Example 2", "Example 3"],
}

# File Upload Configuration
UPLOAD_CONFIG = {
    "allowed_types": ["jpg", "jpeg", "png", "tif", "tiff"],
    "help_text": "Upload a drone image to detect buildings",
}

# UI Styling Configuration
CUSTOM_CSS = """
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
"""

# Application Text Content
APP_CONTENT = {
    "main_title": "🏠 Land Cover Classification",
    "subtitle": "AI-powered building detection from drone imagery",
    "info_box_content": """
    <div class="info-box">
        <h3>How it works:</h3>
        <ul>
            <li>Upload a drone image (JPG, PNG, or TIF format)</li>
            <li>Our AI model will analyze the image and detect buildings</li>
            <li>View the original image, prediction mask, and overlay</li>
            <li>Download the results for further analysis</li>
        </ul>
    </div>
    """,
    "sidebar_about": """
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
        """,
    "sidebar_performance": """
        - **Architecture:** U-Net
        - **Training Data:** Drone imagery patches
        - **Validation Dice Score:** ~0.81
        - **Validation IoU Score:** ~0.68
        """,
    "sidebar_links": """
        - [GitHub Repository](#)
        - [Model Training Notebook](#)
        - [Documentation](#)
        """,
}

# Download Configuration
DOWNLOAD_CONFIG = {
    "chunk_size": 8192,
    "min_file_size": 1000000,  # 1MB minimum
    "timeout": 300,  # 5 minutes timeout
}

# Overlay Configuration
OVERLAY_CONFIG = {
    "building_color": (255, 0, 0, 128),  # Red with transparency
    "mask_threshold": 127,
}
