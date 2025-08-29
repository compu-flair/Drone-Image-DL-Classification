# 🏞️ Land Cover Classification - Streamlit App

A Streamlit web application for building detection from drone imagery using a trained U-Net model.

## 🌐 Live Demo

**Streamlit App:** [🚀 Deploy on Streamlit Cloud](https://yiihuang-drone-image-dl-classification-appstreamlit-app-zhz2xv.streamlit.app/)

**GitHub Repository:** [🔗 Your Repository Link]

### 🚀 Quick Deploy

**Streamlit Cloud (Recommended)**
1. 🍴 Fork this repository
2. 🌐 Go to [share.streamlit.io](https://share.streamlit.io)
3. 🔗 Connect your GitHub account
4. 📂 Select this repository
5. 🗂️ Set main file: `app/streamlit_app.py`
6. 📦 Set requirements: `requirements.txt`
7. 🚀 Deploy!

**Local Development**

Create a virtual environment, install dependencies, and run the app.

```bash
python3 -m venv .venv
```

Then, run the following commands to activate the environment, install dependencies, and start the app.

```bash
# 🟢 Activate virtual environment
source .venv/bin/activate

# 📦 Install dependencies
pip install -r requirements.txt

# ▶️ Run the Streamlit app from the app folder
cd app && streamlit run streamlit_app.py
```

---

## ✨ Features

- 🖼️ **Image Upload**: Drag-and-drop or click-to-upload interface
- 🤖 **AI Processing**: Real-time building detection using trained U-Net model
- 📊 **Visualization**: Display original image, prediction mask, and overlay
- 💾 **Download Results**: Save processed images for further analysis
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Cached Processing**: Example images processed once for better performance

## 🛠️ Prerequisites

- 🐍 Python 3.8 or higher
- 🧠 Trained U-Net model (automatically downloaded from Google Drive)
- 💻 CUDA-compatible GPU (optional, for faster inference)

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Drone-Image-DL-Classification
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Streamlit App

1. **Start the Streamlit application**
   ```bash
   cd app && streamlit run streamlit_app.py
   ```

2. **Open your web browser**
   🌍 Navigate to: `http://localhost:8501`

3. **Upload an image**
   - 🖼️ Drag and drop an image file, or
   - 📁 Click "Choose File" to browse and select an image
   - 🏷️ Supported formats: JPG, PNG, TIF
   - 📏 Maximum file size: 16MB

## ⚙️ How It Works

1. 🧠 **Model Loading**: The app automatically downloads the pre-trained U-Net model from Google Drive on first run
2. ⚡ **Example Processing**: Example images are processed once and cached for better performance
3. 🖼️ **Image Upload**: User uploads a drone image through the web interface
4. 🧹 **Preprocessing**: Image is resized to 512x512 pixels and normalized (0-1 range)
5. 🤖 **Model Inference**: Preprocessed image is fed through the trained U-Net model
6. 🏗️ **Post-processing**: Model output is converted to binary mask (buildings vs. background)
7. 📊 **Visualization**: Results are displayed as:
   - 🖼️ Original image
   - 🏢 Building detection mask (white = buildings, black = background)
   - 🖌️ Overlay view (original image with red building overlay)

## 📁 Project Structure

```
Drone-Image-DL-Classification/
├── app/                       # Main application code
│   ├── __init__.py            # Python package initializer
│   ├── best_unet_model.pth    # Trained U-Net model weights
│   ├── config.py              # App configuration settings
│   ├── data_setup.py          # Data preprocessing utilities
│   ├── lightning_unet.py      # PyTorch Lightning U-Net implementation
│   ├── streamlit_app.py       # Main Streamlit web app
│   ├── unet_model.py          # U-Net model architecture (vanilla)
│   ├── img/                   # Example and UI images
│   │   ├── 0_mlflow_homepage.png   # MLflow UI screenshot
│   │   ├── 1_mlflow_experimenthome.png # MLflow experiment screenshot
│   │   ├── 2_compare_start.png       # Comparison UI screenshot
│   │   ├── 3_compare.png             # Comparison UI screenshot
│   │   ├── 4_individual.png          # Individual result screenshot
│   │   ├── 5_individual_overview.png # Individual overview screenshot
│   │   ├── 6_ngrok.png               # Ngrok tunnel screenshot
│   │   ├── example_1.tif             # Example drone image
│   │   ├── example_2.tif             # Example drone image
│   │   └── example_3.tif             # Example drone image
│   └── pages/
│       └── About.py              # About page for Streamlit app
├── data/                       # Data folders for raw, interim, and processed data
│   ├── 0_raw/                  # Raw input data (e.g., original images, geojson)
│   ├── 1_interim/              # Interim processed data (e.g., masks, RGB)
│   └── 2_patches/              # Image and label patches for training
├── nb/                         # Notebooks and experiment logs
│   ├── best_unet_model.pth     # Trained model weights (notebook context)
│   ├── data/                   # Notebook-specific data folders
│   │   ├── 0_raw/              # Raw data for notebooks
│   │   ├── 1_interim/          # Interim data for notebooks
│   │   └── 2_patches/          # Patches for notebooks
│   ├── local/                  # Main notebook experiments
│   │   ├── 1_dataprep.ipynb    # Data preparation notebook
│   │   ├── 2_model-training.ipynb # Model training notebook
│   │   ├── 3_experiment.ipynb  # Experiment and analysis notebook
│   │   ├── best_unet_model.pth # Trained model weights (local)
│   │   ├── data/               # Data for local notebooks
│   │   ├── lightning_logs/     # PyTorch Lightning logs
│   │   └── mlruns/             # MLflow experiment logs
│   └── party-time/             # Alternate notebook experiments
│       ├── 1_dataprep.ipynb    # Data preparation notebook (alt)
│       ├── 2_model-training.ipynb # Model training notebook (alt)
│       └── 3_experiment.ipynb  # Experiment notebook (alt)
├── format_code.py              # Utility for code formatting
├── pyproject.toml              # Project metadata and build config
├── requirements.txt            # Python dependencies
├── LICENSE                     # License information
├── README.md                   # Project documentation
├── contribution.md             # Contribution guidelines
```

## 🧠 Model Information

- 🏗️ **Architecture**: U-Net with skip connections
- 🖼️ **Input**: RGB images (512x512 pixels)
- 🏢 **Output**: Binary building masks
- 🔥 **Framework**: PyTorch
- 🗂️ **Training Data**: Drone imagery patches
- 📈 **Performance**: Validation Dice Score ~0.81, IoU Score ~0.68

## ⚙️ Configuration

The application can be configured by modifying settings in `app/config.py`:

- 🏗️ **Model configuration**: File path, Google Drive ID, input/output channels
- 🎨 **UI settings**: Page title, styling, content text
- 📁 **File upload**: Supported formats and validation
- 🖼️ **Example images**: Paths and captions
- ⚡ **Processing parameters**: Image size, thresholds, overlay colors

## 🆘 Troubleshooting

### 🐞 Common Issues

1. **Model download fails**
   ```
   Error: Failed to download model file
   ```
   💡 **Solution**: Check internet connection and Google Drive access

2. **Import errors**
   ```
   ModuleNotFoundError: No module named 'unet_model'
   ```
   💡 **Solution**: Ensure you're running from the app directory: `cd app && streamlit run streamlit_app.py`

3. **CUDA out of memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   💡 **Solution**: The app will automatically fall back to CPU if GPU memory is insufficient

4. **Port already in use**
   ```
   OSError: [Errno 98] Address already in use
   ```
   💡 **Solution**: Kill existing Streamlit processes or use a different port

### ⚡ Performance Tips

- 🖥️ **GPU Usage**: The app automatically uses CUDA if available for faster inference
- 🗃️ **Caching**: Example images are processed once and cached for better performance
- 📏 **Image Size**: Larger images take longer to process due to resizing
- 🧠 **Memory**: Close other applications if experiencing memory issues

## 🛠️ Development

### 🧹 Code Formatting

This project uses automated code formatting to maintain consistent style:

**Automatic Formatting:**
```bash
# 🧹 Format all Python files (includes EOF newline checks)
python format_code.py

# Or run individually:
source .venv/bin/activate && black *.py app/*.py          # Code formatting
source .venv/bin/activate && isort *.py app/*.py          # Import organization
```

**Manual Formatting:**
```bash
# 🧹 Format specific files
source .venv/bin/activate && black app/streamlit_app.py app/unet_model.py
source .venv/bin/activate && isort app/streamlit_app.py app/unet_model.py
```

### ✨ Adding New Features

1. 🏗️ **New Model Architecture**: Update the model loading in `app/streamlit_app.py`
2. 🧹 **Additional Preprocessing**: Modify the `preprocess_image()` function in `app/streamlit_app.py`
3. 🎨 **UI Enhancements**: Edit the Streamlit interface in `app/streamlit_app.py`
4. ⚙️ **Configuration Changes**: Modify settings in `app/config.py`

### 🧪 Testing

Test the model functionality:
```bash
# ▶️ Run the Streamlit app and test with example images
cd app && streamlit run streamlit_app.py
```

## 📦 Dependencies

Key dependencies include:
- 🟢 **Streamlit**: Web application framework
- 🔥 **PyTorch**: Deep learning framework
- 🖼️ **Pillow**: Image processing
- 🔢 **NumPy**: Numerical computing
- 🧬 **Albumentations**: Image augmentation
- 🌐 **Requests**: HTTP requests for model download

## 📄 License

This project is part of the Land Cover Classification system. See the LICENSE file for details.

## 💬 Support

For issues and questions:
1. 🆘 Check the troubleshooting section above
2. 📦 Ensure all dependencies are installed correctly
3. 📝 Check the Streamlit app logs for error messages
4. 📁 Verify the model file is accessible

## 🤝 Contributing

1. 🍴 Fork the repository
2. 🌱 Create a feature branch
3. 🛠️ Make your changes
4. 🧪 Test thoroughly
5. 📤 Submit a pull request 
