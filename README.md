# Land Cover Classification Web App

A Flask-based web application for building detection from drone imagery using a trained U-Net model.

## 🌐 Live Demo

**Streamlit App:** [Deploy on Streamlit Cloud](https://share.streamlit.io)

**Flask App:** [Deploy on Heroku/Railway/Vercel]

**GitHub Repository:** [Your Repository Link]

### 🚀 Quick Deploy

**Option 1: Streamlit Cloud (Recommended)**
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file: `webapp/streamlit_app.py`
6. Set requirements: `webapp/requirements_streamlit.txt`
7. Deploy!

**Option 2: Local Development**
```bash
cd webapp
source ../.venv/bin/activate
pip install -r requirements.txt
python app.py
```

---

## Features

- 🖼️ **Image Upload**: Drag-and-drop or click-to-upload interface
- 🤖 **AI Processing**: Real-time building detection using trained U-Net model
- 📊 **Visualization**: Display original image, prediction mask, and overlay
- 💾 **Download Results**: Save processed images for further analysis
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.8 or higher
- Trained U-Net model (`best_unet_model.pth`) in the parent directory
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd webapp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify the trained model exists**
   ```bash
   ls ../Drone-Image-DL-Classification/best_unet_model.pth
   ```

## Testing the Setup

Before running the web app, test that all components work correctly:

```bash
python test_model.py
```

This will test:
- ✅ Model loading
- ✅ Preprocessing pipeline
- ✅ Inference pipeline

## Running the Web App

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Open your web browser**
   Navigate to: `http://localhost:5000`

3. **Upload an image**
   - Drag and drop an image file, or
   - Click "Choose File" to browse and select an image
   - Supported formats: JPG, PNG, TIF
   - Maximum file size: 16MB

## How It Works

1. **Image Upload**: User uploads a drone image through the web interface
2. **Preprocessing**: Image is resized to 512x512 pixels and normalized (0-1 range)
3. **Model Inference**: Preprocessed image is fed through the trained U-Net model
4. **Post-processing**: Model output is converted to binary mask (buildings vs. background)
5. **Visualization**: Results are displayed as:
   - Original image
   - Building detection mask (white = buildings, black = background)
   - Overlay view (original image with red building overlay)

## Project Structure

```
webapp/
├── app.py                 # Main Flask application
├── test_model.py          # Model testing script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface template
├── uploads/              # Temporary storage for uploaded files
└── results/              # Storage for processed results
```

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Image upload and processing endpoint
- `GET /health` - Health check endpoint

## Configuration

The application can be configured by modifying variables in `app.py`:

- `MAX_CONTENT_LENGTH`: Maximum file upload size (default: 16MB)
- `UPLOAD_FOLDER`: Directory for uploaded files
- `RESULTS_FOLDER`: Directory for processed results

## Troubleshooting

### Common Issues

1. **Model file not found**
   ```
   Error: Model file not found at ../Drone-Image-DL-Classification/best_unet_model.pth
   ```
   **Solution**: Ensure the trained model file exists in the correct location

2. **Import errors**
   ```
   ModuleNotFoundError: No module named 'unet_model'
   ```
   **Solution**: Check that the path to the model source files is correct

3. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: The app will automatically fall back to CPU if GPU memory is insufficient

4. **Port already in use**
   ```
   OSError: [Errno 98] Address already in use
   ```
   **Solution**: Change the port in `app.py` or kill the existing process

### Performance Tips

- **GPU Usage**: The app automatically uses CUDA if available for faster inference
- **Image Size**: Larger images take longer to process due to resizing
- **Memory**: Close other applications if experiencing memory issues

## Development

### Adding New Features

1. **New Model Architecture**: Update the model loading in `app.py`
2. **Additional Preprocessing**: Modify the `process_image()` function
3. **UI Enhancements**: Edit `templates/index.html`

### Testing

Run the test suite:
```bash
python test_model.py
```

### Logging

The application logs important events. Check the console output for:
- Model loading status
- Processing errors
- Performance metrics

## License

This project is part of the Land Cover Classification system. See the main project README for license information.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run the test script to verify setup
3. Check the console logs for error messages
4. Ensure all dependencies are installed correctly 