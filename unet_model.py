import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class DoubleConv(nn.Module):
    """
    Double Convolution block: (Conv2d -> BatchNorm -> ReLU) x 2
    This is used repeatedly throughout the U-Net architecture.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            # padding=1 to preserve spatial dimensions while extracting features 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            # BatchNorm is added to help with training stability and potentially faster convergence
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownSamplingBlock(nn.Module):
    """
    Downsampling block for the encoder path: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(DownSamplingBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class UpSamplingBlock(nn.Module):
    """
    Upsampling block for the decoder path: 
    Upsample -> Concatenate with skip connection -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(UpSamplingBlock, self).__init__()
        # Transposed conv for upsampling
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, input channels will be doubled
        self.double_conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = self.up(x)
        
        # Make sure dimensions match for concatenation
        # Calculate padding if needed
        diff_y = skip_connection.size()[2] - x.size()[2]
        diff_x = skip_connection.size()[3] - x.size()[3]
        
        # Pad if needed
        if diff_y > 0 or diff_x > 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([skip_connection, x], dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    """
    Complete U-Net architecture using modular building blocks.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Initial double convolution
        self.initial_conv = DoubleConv(in_channels, 64)
        
        # Encoder (Downsampling) path
        self.down1 = DownSamplingBlock(64, 128)
        self.down2 = DownSamplingBlock(128, 256)
        self.down3 = DownSamplingBlock(256, 512)
        self.down4 = DownSamplingBlock(512, 1024)
        
        # Decoder (Upsampling) path
        self.up1 = UpSamplingBlock(1024, 512)
        self.up2 = UpSamplingBlock(512, 256)
        self.up3 = UpSamplingBlock(256, 128)
        self.up4 = UpSamplingBlock(128, 64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encode
        x1 = self.initial_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decode with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final 1x1 convolution
        output = self.out_conv(x)
        
        return output

# Define metrics functions
class Metrics:
    @staticmethod
    def dice_coefficient(y_pred, y_true, smooth=1e-6):
        """
        Calculate Dice coefficient
        
        Args:
            y_pred: Predicted masks, after sigmoid (B, 1, H, W)
            y_true: Ground truth masks (B, 1, H, W)
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient (0-1, higher is better)
        """
        # Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
        
        return dice.item()

    @staticmethod
    def iou_score(y_pred, y_true, smooth=1e-6):
        """
        Calculate IoU score (Jaccard index)
        
        Args:
            y_pred: Predicted masks, after sigmoid (B, 1, H, W)
            y_true: Ground truth masks (B, 1, H, W)
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            IoU score (0-1, higher is better)
        """
        # Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        intersection = (y_pred * y_true).sum()
        total = (y_pred + y_true).sum()
        union = total - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()

class UNetTrainer:
    def __init__(self, model, train_loader, val_loader, criterion=None, optimizer=None, 
                 device=None, learning_rate=0.001):
        """
        Initialize the UNet trainer
        
        Args:
            model: UNet model instance
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (defaults to BCEWithLogitsLoss)
            optimizer: Optimizer (defaults to Adam)
            device: Device to use (defaults to GPU if available)
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Use GPU if available
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Default loss function
        self.criterion = criterion if criterion else nn.BCEWithLogitsLoss()
        
        # Default optimizer
        self.optimizer = optimizer if optimizer else optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        
        # History for tracking metrics
        self.history = {
            'train_loss': [], 'train_dice': [], 'train_iou': [],
            'val_loss': [], 'val_dice': [], 'val_iou': []
        }
        
        # Best validation score tracking
        self.best_val_dice = 0.0
        
    def train_epoch(self):
        """Run one training epoch"""
        self.model.train()
        epoch_loss = 0
        dice_scores = []
        iou_scores = []
        
        for images, masks in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Add channel dimension to masks if needed
            if masks.dim() == 3:  # [B, H, W]
                masks = masks.unsqueeze(1)  # [B, 1, H, W]
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            
            # Calculate metrics (convert logits to probabilities)
            with torch.no_grad():
                pred_masks = torch.sigmoid(outputs) > 0.5
                pred_masks = pred_masks.float()
                dice_scores.append(Metrics.dice_coefficient(pred_masks, masks))
                iou_scores.append(Metrics.iou_score(pred_masks, masks))
        
        # Calculate average metrics
        avg_loss = epoch_loss / len(self.train_loader)
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        
        return avg_loss, avg_dice, avg_iou
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        dice_scores = []
        iou_scores = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Add channel dimension to masks if needed
                if masks.dim() == 3:  # [B, H, W]
                    masks = masks.unsqueeze(1)  # [B, 1, H, W]

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Track metrics
                val_loss += loss.item()
                
                # Calculate metrics
                pred_masks = torch.sigmoid(outputs) > 0.5
                pred_masks = pred_masks.float()
                dice_scores.append(Metrics.dice_coefficient(pred_masks, masks))
                iou_scores.append(Metrics.iou_score(pred_masks, masks))
        
        # Calculate average metrics
        avg_loss = val_loss / len(self.val_loader)
        avg_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        
        return avg_loss, avg_dice, avg_iou
    
    def train(self, num_epochs=5, save_path='best_unet_model.pth'):
        """Train the model for the specified number of epochs"""
        print("Starting training...")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_dice, train_iou = self.train_epoch()
            
            # Validate
            val_loss, val_dice, val_iou = self.validate()
            
            # Print metrics
            print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
            print(f"Valid - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_dice'].append(train_dice)
            self.history['train_iou'].append(train_iou)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            
            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                torch.save(self.model.state_dict(), save_path)
                print(f"Model saved with Dice score: {val_dice:.4f}")
        
        print("Training complete!")
        return self.history
        