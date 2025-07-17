# Deployment Guide

This guide covers different deployment options for the Land Cover Classification web app.

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

**Pros:**
- Free hosting
- Easy deployment
- Automatic HTTPS
- Built-in CDN
- No server management

**Steps:**
1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `webapp/streamlit_app.py`
   - Set requirements file: `webapp/requirements_streamlit.txt`
   - Click "Deploy"

3. **Your app will be available at:**
   ```
   https://your-app-name.streamlit.app
   ```

### Option 2: Heroku (Flask)

**Pros:**
- Free tier available
- Easy deployment
- Automatic scaling

**Steps:**
1. **Create Procfile:**
   ```
   web: gunicorn app:app
   ```

2. **Update requirements.txt:**
   ```
   gunicorn>=20.1.0
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Railway (Flask)

**Pros:**
- Free tier available
- Easy deployment
- Automatic HTTPS

**Steps:**
1. **Connect GitHub repository**
2. **Set environment variables:**
   - `PORT=5000`
3. **Deploy automatically**

### Option 4: Vercel (Flask)

**Pros:**
- Free tier available
- Fast deployment
- Global CDN

**Steps:**
1. **Create vercel.json:**
   ```json
   {
     "version": 2,
     "builds": [
       {
         "src": "webapp/app.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "webapp/app.py"
       }
     ]
   }
   ```

2. **Deploy:**
   ```bash
   vercel
   ```

## 📁 File Structure for Deployment

```
Land_Cover_Classification/
├── webapp/
│   ├── streamlit_app.py          # Streamlit version
│   ├── app.py                    # Flask version
│   ├── requirements_streamlit.txt # Streamlit dependencies
│   ├── requirements.txt          # Flask dependencies
│   ├── templates/
│   │   └── index.html
│   └── .gitignore
├── src/
│   ├── unet_model.py
│   └── data_setup.py
└── best_unet_model.pth           # Trained model
```

## 🔧 Model File Handling

### For Streamlit Cloud:
- Include the model file in your repository
- Update the path in `streamlit_app.py`:
  ```python
  model_path = 'best_unet_model.pth'  # Relative to app directory
  ```

### For Other Platforms:
- Consider using cloud storage (AWS S3, Google Cloud Storage)
- Update the model loading to download from URL:
  ```python
  import requests
  
  def download_model():
      url = "https://your-storage-url/best_unet_model.pth"
      response = requests.get(url)
      with open("best_unet_model.pth", "wb") as f:
          f.write(response.content)
  ```

## 🌐 Custom Domain (Optional)

### For Streamlit:
- Not directly supported
- Use reverse proxy (Cloudflare, Nginx)

### For Flask:
- Add custom domain in platform settings
- Update DNS records

## 📊 Monitoring and Analytics

### Streamlit:
- Built-in analytics dashboard
- Usage statistics available

### Flask:
- Add logging:
  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

## 🔒 Security Considerations

1. **File Upload Limits:**
   ```python
   app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
   ```

2. **Input Validation:**
   ```python
   ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
   ```

3. **Environment Variables:**
   ```python
   import os
   SECRET_KEY = os.environ.get('SECRET_KEY', 'default-key')
   ```

## 🚨 Troubleshooting

### Common Issues:

1. **Model Loading Error:**
   - Check file path
   - Ensure model file is included in deployment

2. **Memory Issues:**
   - Reduce model size
   - Use CPU-only deployment
   - Optimize image preprocessing

3. **Import Errors:**
   - Check requirements.txt
   - Verify all dependencies are listed

4. **Port Issues:**
   - Use environment variable for port:
     ```python
     port = int(os.environ.get('PORT', 5000))
     ```

## 📈 Performance Optimization

1. **Model Caching:**
   ```python
   @st.cache_resource
   def load_model():
       # Model loading code
   ```

2. **Image Optimization:**
   - Resize images before processing
   - Use efficient image formats

3. **Memory Management:**
   - Clear cache periodically
   - Use generators for large datasets

## 🔄 CI/CD Pipeline

### GitHub Actions Example:
```yaml
name: Deploy to Streamlit
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to Streamlit Cloud
      uses: streamlit/streamlit-deploy-action@v0.1.0
      with:
        streamlit_app_file: webapp/streamlit_app.py
        requirements_file: webapp/requirements_streamlit.txt
```

## 📞 Support

For deployment issues:
1. Check platform-specific documentation
2. Review error logs
3. Test locally first
4. Use platform support channels 