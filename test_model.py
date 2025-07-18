#!/usr/bin/env python3
"""
Test script to verify model loading and preprocessing pipeline
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

# Add the model source path
sys.path.append("../Drone-Image-DL-Classification/src")
from data_setup import get_preprocessing
from unet_model import UNet


def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Initialize model
        model = UNet(in_channels=3, out_channels=1)

        # Load trained weights
        model_path = "../Drone-Image-DL-Classification/best_unet_model.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            print("✅ Model loaded successfully")
            return True
        else:
            print(f"❌ Model file not found at {model_path}")
            return False

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def test_preprocessing():
    """Test the preprocessing pipeline"""
    print("\nTesting preprocessing pipeline...")

    try:
        # Initialize preprocessing
        preprocessing = get_preprocessing()

        # Create a dummy image (512x512 RGB)
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        dummy_mask = np.zeros((512, 512), dtype=np.float32)

        # Apply preprocessing
        sample = preprocessing(image=dummy_image, mask=dummy_mask)
        processed_image = sample["image"]

        # Check output shape
        expected_shape = (3, 512, 512)
        if processed_image.shape == expected_shape:
            print("✅ Preprocessing pipeline works correctly")
            print(f"   Input shape: {dummy_image.shape}")
            print(f"   Output shape: {processed_image.shape}")
            return True
        else:
            print(
                f"❌ Unexpected output shape: {processed_image.shape}, expected: {expected_shape}"
            )
            return False

    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        return False


def test_inference_pipeline():
    """Test the complete inference pipeline"""
    print("\nTesting inference pipeline...")

    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        model = UNet(in_channels=3, out_channels=1)
        model_path = "../Drone-Image-DL-Classification/best_unet_model.pth"

        if not os.path.exists(model_path):
            print("❌ Model file not found, skipping inference test")
            return False

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Initialize preprocessing
        preprocessing = get_preprocessing()

        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Preprocess
        sample = preprocessing(image=dummy_image, mask=np.zeros((512, 512)))
        processed_image = sample["image"]

        # Convert to tensor
        input_tensor = torch.from_numpy(processed_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output) > 0.5
            prediction = prediction.cpu().numpy()[0, 0]

        print("✅ Inference pipeline works correctly")
        print(f"   Input tensor shape: {input_tensor.shape}")
        print(f"   Output shape: {prediction.shape}")
        print(f"   Prediction range: [{prediction.min()}, {prediction.max()}]")
        return True

    except Exception as e:
        print(f"❌ Error in inference pipeline: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Land Cover Classification Web App Components")
    print("=" * 50)

    # Test model loading
    model_ok = test_model_loading()

    # Test preprocessing
    preprocessing_ok = test_preprocessing()

    # Test inference pipeline
    inference_ok = test_inference_pipeline()

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)
    print(f"Model Loading: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Preprocessing: {'✅ PASS' if preprocessing_ok else '❌ FAIL'}")
    print(f"Inference Pipeline: {'✅ PASS' if inference_ok else '❌ FAIL'}")

    if all([model_ok, preprocessing_ok, inference_ok]):
        print("\n🎉 All tests passed! The web app should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return all([model_ok, preprocessing_ok, inference_ok])


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
