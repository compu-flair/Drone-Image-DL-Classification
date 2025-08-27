import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex

# Import the original UNet model - this assumes unet_model.py is in the same directory
from .unet_model import UNet, Metrics

class LightningUNet(L.LightningModule):
    """
    PyTorch Lightning wrapper for the UNet model.
    This can be used in a Jupyter notebook without modifying the original UNet implementation.
    """
    def __init__(self, in_channels=3, out_channels=1, learning_rate=0.001, 
                 use_binary_threshold=True, threshold=0.5):
        """
        Initialize the Lightning UNet model
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB images)
            out_channels: Number of output channels (default: 1 for binary segmentation)
            learning_rate: Learning rate for the optimizer (default: 0.001)
            use_binary_threshold: Whether to apply a threshold to get binary masks (default: True)
            threshold: Threshold value for binary classification (default: 0.5)
        """
        super().__init__()
        
        # Create an instance of the original UNet model
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)
        
        # Parameters
        self.learning_rate = learning_rate
        self.use_binary_threshold = use_binary_threshold
        self.threshold = threshold
        
        # Loss function - same as the original implementation
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics - using torchmetrics for better integration with Lightning
        self.dice_metric = DiceScore(average='micro', num_classes=2)
        self.iou_metric = JaccardIndex(task='binary', num_classes=2)
        
        # Metrics for each stage
        self.train_dice = DiceScore(average='micro', num_classes=2)
        self.val_dice = DiceScore(average='micro', num_classes=2)
        self.test_dice = DiceScore(average='micro', num_classes=2)
        
        self.train_iou = JaccardIndex(task='binary', num_classes=2)
        self.val_iou = JaccardIndex(task='binary', num_classes=2)
        self.test_iou = JaccardIndex(task='binary', num_classes=2)
        
        # Save hyperparameters for easier model loading
        self.save_hyperparameters(ignore=['unet'])
    
    def forward(self, x):
        """Forward pass through the UNet model"""
        return self.unet(x)
    
    def _shared_step(self, batch, stage):
        """Shared step for training, validation, and testing"""
        images, masks = batch
        
        # Add channel dimension to masks if needed
        if masks.dim() == 3:  # [B, H, W]
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.criterion(logits, masks)
        
        # Get predictions
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float() if self.use_binary_threshold else probs
        
        # Track metrics based on stage
        if stage == 'train':
            self.train_dice(preds, masks)
            self.train_iou(preds.squeeze(1), masks.squeeze(1).long())
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_dice', self.train_dice, prog_bar=True)
            self.log('train_iou', self.train_iou, prog_bar=True)
        elif stage == 'val':
            self.val_dice(preds, masks)
            self.val_iou(preds.squeeze(1), masks.squeeze(1).long())
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_dice', self.val_dice, prog_bar=True)
            self.log('val_iou', self.val_iou, prog_bar=True)
        elif stage == 'test':
            self.test_dice(preds, masks)
            self.test_iou(preds.squeeze(1), masks.squeeze(1).long())
            self.log('test_loss', loss)
            self.log('test_dice', self.test_dice)
            self.log('test_iou', self.test_iou)
            
        return loss
    
    def training_step(self, batch, batch_idx):
        """Single training step"""
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        """Single validation step"""
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        """Single test step"""
        return self._shared_step(batch, 'test')
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler - reduce on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5, 
            # verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }
    
    def predict_segmentation(self, image_tensor):
        """
        Generate a segmentation mask for a single image
        
        Args:
            image_tensor: Image tensor with shape [C, H, W] or [1, C, H, W]
            
        Returns:
            Binary segmentation mask
        """
        self.eval()
        
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self(image_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > self.threshold).float() if self.use_binary_threshold else probs
            
        return mask.squeeze(0)  # Remove batch dimension

# Utility functions to use in a Jupyter notebook
def train_lightning_unet(model, train_loader, val_loader, max_epochs=10, 
                         gpus=None, precision=32, callbacks=None, progress_bar_refresh_rate=10):
    """
    Train a LightningUNet model
    
    Args:
        model: LightningUNet model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        max_epochs: Maximum number of epochs (default: 10)
        gpus: Number of GPUs to use (default: None, will use 1 if available)
        precision: Precision for training (default: 32)
        callbacks: List of callbacks (default: None)
        progress_bar_refresh_rate: Progress bar refresh rate (default: 10)
        
    Returns:
        Trained model and trainer
    """
    # Default callbacks if none provided
    if callbacks is None:
        callbacks = [
            L.callbacks.ModelCheckpoint(
                monitor='val_dice',
                mode='max',
                save_top_k=1,
                filename='best-{epoch:02d}-{val_dice:.4f}',
                verbose=True
            ),
            L.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min',
                verbose=True
            ),
            L.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
    
    # Configure trainer
    if gpus is None:
        gpus = 1 if torch.cuda.is_available() else 0
        
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus,
        callbacks=callbacks,
        log_every_n_steps=progress_bar_refresh_rate,
        precision=precision
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    return model, trainer

def evaluate_lightning_unet(model, test_loader):
    """
    Evaluate a trained LightningUNet model
    
    Args:
        model: Trained LightningUNet model
        test_loader: Test data loader
    
    Returns:
        Test results dictionary
    """
    trainer = L.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    results = trainer.test(model, test_loader, verbose=True)
    
    return results
