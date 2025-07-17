import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio as rio
import matplotlib.pyplot as plt
from random import randint
import albumentations as A
from sklearn.model_selection import train_test_split

class BuildingsDroneDataset(Dataset):
    """Dataset for building segmentation from drone imagery.

    Args:
        patch_dir (str): path to patch images folder (TIF format)
        label_dir (str): path to label masks folder (TIF format)
        augmentation (albumentations.Compose): data transformation pipeline
        preprocessing (albumentations.Compose): data preprocessing
    """

    def __init__(
        self,
        patch_dir,
        label_dir,
        augmentation=None,
        preprocessing=None,
    ):
        # Get paths to all TIF files in the directories
        self.patch_paths = [os.path.join(patch_dir, file_name) 
                          for file_name in sorted(os.listdir(patch_dir)) 
                          if file_name.endswith('.tif')]
        
        self.label_paths = [os.path.join(label_dir, file_name) 
                           for file_name in sorted(os.listdir(label_dir)) 
                           if file_name.endswith('.tif')]
        
        # Verify that we have matching number of patches and labels
        assert len(self.patch_paths) == len(self.label_paths), \
            f"Number of patches ({len(self.patch_paths)}) and labels ({len(self.label_paths)}) don't match"
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # Read TIF images using rasterio
        with rio.open(self.patch_paths[i]) as patch_file:
            # Read all bands and transpose to (channels, height, width)
            patch = patch_file.read()
            # Convert to (height, width, channels) for albumentations
            patch = np.transpose(patch, (1, 2, 0))
            
        with rio.open(self.label_paths[i]) as label_file:
            # Assume mask is single channel
            mask = label_file.read(1)
            
        # Ensure mask has values 0 and 1 only
        mask = (mask > 0).astype(np.float32)
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=patch, mask=mask)
            patch, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=patch, mask=mask)
            patch, mask = sample['image'], sample['mask']
            
        # Convert numpy arrays to PyTorch tensors
        patch = torch.from_numpy(patch)
        mask = torch.from_numpy(mask)
            
        return patch, mask

    def __len__(self):
        return len(self.patch_paths)

    def plot_pair(self, idx=None):
        if idx is not None:
            idx = idx
        else:
            idx = randint(0, len(self)-1)
        _, ax = plt.subplots(1, 2, figsize=(8, 4))

        img, mask = self[idx]
        
        # Convert tensors back to numpy for plotting
        img = img.numpy()
        mask = mask.numpy()
        
        # Handle different dimensions based on preprocessing
        if len(img.shape) == 3 and img.shape[0] == 3:  # CHW format
            img = np.transpose(img, (1, 2, 0))  # Convert to HWC for plotting
            
        ax[0].imshow(img)
        ax[0].set_title("Drone Image")
        
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Building Mask")

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
            
        plt.suptitle(f"Sample {idx}")
        plt.tight_layout()

def get_training_augmentation():
    """
    Simple augmentation pipeline for drone imagery training data
    """
    train_transform = [
        # Basic spatial augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    return A.Compose(train_transform)

def get_preprocessing():
    """
    Simple preprocessing function for drone imagery
    
    Returns:
        Preprocessing transform for normalization and channel reordering
    """
    # Define regular functions instead of lambdas
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

def create_dataloaders(patch_dir, label_dir, batch_size=8, val_split=0.2, random_state=42, num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        patch_dir: Directory containing image patches
        label_dir: Directory containing label masks
        batch_size: Batch size for dataloaders
        val_split: Validation split ratio (0.0-1.0)
        random_state: Random seed for reproducibility
        num_workers: Number of workers for DataLoader
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    # Create dataset with augmentation and preprocessing
    dataset = BuildingsDroneDataset(
        patch_dir=patch_dir,
        label_dir=label_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing()
    )
    
    # Split dataset into training and validation
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=val_split,
        random_state=random_state
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader