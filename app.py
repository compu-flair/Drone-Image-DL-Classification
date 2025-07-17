from flask import Flask, render_template, request, jsonify, send_file
import os
import io
import base64
from PIL import Image
import torch
import numpy as np
from werkzeug.utils import secure_filename
import logging

# Import our model and preprocessing
import sys
sys.path.append('../Drone-Image-DL-Classification/src')
from unet_model import UNet
from data_setup import get_preprocessing

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
device = None
preprocessing = None

def load_model():
    """Load the trained UNet model"""
    global model, device, preprocessing
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model
        model = UNet(in_channels=3, out_channels=1)
        
        # Load trained weights
        model_path = '../Drone-Image-DL-Classification/best_unet_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            logger.info("Model loaded successfully")
        else:
            logger.error(f"Model file not found at {model_path}")
            return False
        
        # Initialize preprocessing
        preprocessing = get_preprocessing()
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            result = process_image(filepath)
            
            if result['success']:
                return jsonify(result)
            else:
                return jsonify({'error': result['error']}), 500
                
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

def process_image(image_path):
    """Process uploaded image and return prediction"""
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (512x512)
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply preprocessing (same as training)
        if preprocessing:
            sample = preprocessing(image=image_array, mask=np.zeros((512, 512)))
            processed_image = sample['image']  # Shape: (3, 512, 512)
        else:
            # Fallback preprocessing
            processed_image = image_array.astype(np.float32) / 255.0
            processed_image = processed_image.transpose(2, 0, 1)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output) > 0.5
            prediction = prediction.cpu().numpy()[0, 0]  # Remove batch and channel dims
        
        # Convert prediction to image
        prediction_image = (prediction * 255).astype(np.uint8)
        prediction_pil = Image.fromarray(prediction_image, mode='L')
        
        # Create overlay
        overlay = create_overlay(image, prediction_pil)
        
        # Convert images to base64 for web display
        original_b64 = image_to_base64(image)
        prediction_b64 = image_to_base64(prediction_pil)
        overlay_b64 = image_to_base64(overlay)
        
        return {
            'success': True,
            'original': original_b64,
            'prediction': prediction_b64,
            'overlay': overlay_b64,
            'filename': os.path.basename(image_path)
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {'success': False, 'error': str(e)}

def create_overlay(original_image, prediction_image):
    """Create overlay of original image with prediction mask"""
    # Convert prediction to RGBA with red overlay
    overlay = original_image.copy()
    prediction_rgba = prediction_image.convert('RGBA')
    
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

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Web app starting successfully")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("Failed to load model. Exiting.")
        exit(1) 