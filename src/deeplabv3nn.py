# Copyright 2024 The MathWorks, Inc.

# Uncomment the following lines to enable debugging
# import debugpy
# debugpy.debug_this_thread()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101
import warnings
import numpy as np

def load_model(backbone="resnet50",num_classes:int=3):
    # Load the DeepLabV3 model with the specified backbone
    if backbone == "resnet50":
        model = deeplabv3_resnet50(weights='DEFAULT')
    elif backbone == "resnet101":
        model = deeplabv3_resnet101(weights='DEFAULT')

    # Modify the classifier to output the desired number of classes
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    
    return model

def analyze_model(backbone,num_classes:int):
    # Load the model with the specified backbone
    model = load_model(backbone=backbone, num_classes=num_classes)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count the number of modules (layers)
    num_layers = len(list(model.modules()))

    return [num_layers, total_params]

def save_checkpoint(state,filename="saved_model.pth"):
    # Save the model checkpoint
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Define a custom loss class combining Dice coefficient and Binary Cross Entropy loss
class diceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(diceBCELoss, self).__init__()
        self.bce_losss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1):
        # Compute Binary Cross Entropy loss
        BCE = self.bce_losss(inputs, targets)

        # Apply sigmoid activation to inputs if needed
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute Dice loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        # Combine BCE and Dice loss
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

# Define a custom loss class for Intersection over Union (IoU)
class IOU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IOU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Apply sigmoid activation to inputs if needed
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        # Compute IoU
        IoU = (intersection + smooth) / (union + smooth)

        return IoU

class trainer:
    # Constructor method to initialize the object
    def __init__(self, network_backbone,
                 train_now, 
                 file_name, 
                 save_chk_pt_freq=100, 
                 max_train_iter=500):
        """
        Initializes the trainer class.

        Parameters:
        - network_backbone: The backbone architecture for the network.
        - train_now: Boolean indicating whether to initialize for training.
        - file_name: The file name to save or load the model checkpoint.
        - save_chk_pt_freq: Frequency of saving checkpoints during training.
        - max_train_iter: Maximum number of training iterations.
        """
        # Set device to GPU if available, otherwise CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if train_now:
            # Load the model with the specified backbone for training
            self.model = load_model(backbone=network_backbone)
            self.save_chk_pt_freq = save_chk_pt_freq
            self.learn_rate = 1e-4
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learn_rate)
            self.loss_function = diceBCELoss()
            self.train_iteration = 0
            self.loss_vector = []
            self.output_loss = []
            self.val_accuracy = []
            self.val_dice_loss = []
            self.num_correct = 0
            self.num_pixels = 0
            self.dice_loss = 0
            self.max_train_iter = max_train_iter
            self.file_name = file_name
        else:
            # Load the model from a checkpoint for inference
            self.model = load_model(backbone=network_backbone)
            self.model.load_state_dict(torch.load(file_name, weights_only=True, map_location=torch.device(self.device)))
        
        # Initialize metrics and device settings
        self.output_function = IOU()
        self.num_correct_test = 0
        self.num_pixels_test = 0
        self.dice_score_test = 0
        self.iou = 0
        self.test_iou = 0
        self.test_accuracy = 0
        self.test_dice_score = 0
        self.testIter = 0

        self.model = self.model.to(self.device)

        warnings.filterwarnings("ignore", message="The given NumPy array is not writable, and PyTorch does not support non-writable tensors")

    # Forward and backward pass, update the model weights and loss functions
    def train_step(self, image_tensor, mask_tensor):
        """
        Performs a single training step.

        Parameters:
        - image_tensor: Input images as a tensor.
        - mask_tensor: Ground truth masks as a tensor.
        """
        self.model.train()

        # Use mixed precision if supported by the device
        with torch.amp.autocast(self.device):
            predictions = self.model(image_tensor)['out']
            loss = self.loss_function(predictions, mask_tensor)
            iou = self.output_function(predictions, mask_tensor)

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_iteration += 1

        # Record loss and IOU
        self.loss_vector.append(loss.item())
        self.output_loss.append(iou.item())

        # Save model checkpoint at specified frequency
        if self.train_iteration % self.save_chk_pt_freq == 0:
            checkpoint = {
                "state_dict": self.model.state_dict(), 
                "optimizer": self.optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=f"saved_model_iter_{int(self.train_iteration)}.pth")

        # Save the final model at the end of training
        if self.train_iteration % self.max_train_iter == 0:
            torch.save(self.model.state_dict(), self.file_name)

    # Compute pixel accuracy and dice score on validation data
    def validate_step(self, image_tensor, mask_tensor):
        """
        Performs a validation step.

        Parameters:
        - image_tensor: Input images as a tensor.
        - mask_tensor: Ground truth masks as a tensor.
        """
        self.model.eval()

        with torch.no_grad():
            preds = torch.sigmoid(self.model(image_tensor)['out'])
            
            # Calculate dice loss
            self.dice_loss += 1 - (2 * (preds * mask_tensor).sum()+1) / (
                (preds + mask_tensor).sum() + 1
            )

            # Calculate pixel accuracy
            preds = (preds > 0.5).float()
            self.num_correct += (preds == mask_tensor).sum()
            self.num_pixels += torch.numel(preds)

        # Record accuracy and dice score
        self.val_accuracy.append(self.num_correct.item() / self.num_pixels * 100)
        self.val_dice_loss.append(self.dice_loss.item())

    def test_step(self, image_tensor, mask_tensor):
        """
        Performs a testing step to evaluate the model on test data.

        Parameters:
        - image_tensor: Input images as a tensor.
        - mask_tensor: Ground truth masks as a tensor.
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Disable gradient calculation for efficiency
        with torch.no_grad():
            # Get model predictions
            predictions = self.model(image_tensor)['out']
            # Apply sigmoid activation to convert logits to probabilities
            preds = torch.sigmoid(predictions)
            # Convert probabilities to binary predictions
            preds = (preds > 0.5).float()
            
            # Calculate the number of correct predictions
            self.num_correct_test += (preds == mask_tensor).sum()
            # Count the total number of pixels
            self.num_pixels_test += torch.numel(preds)
            # Calculate the dice score
            self.dice_score_test += (2 * (preds * mask_tensor).sum()) / (
                (preds + mask_tensor).sum() + 1e-8
            )
            # Calculate the Intersection over Union (IoU) score
            iou = self.output_function(predictions, mask_tensor)
            self.iou += iou.item()

        # Increment the test iteration counter
        self.testIter += 1

        # Calculate average metrics over all test iterations
        self.test_iou = self.iou / self.testIter
        self.test_accuracy = self.num_correct_test.item() / self.num_pixels_test * 100
        self.test_dice_score = self.dice_score_test.item() / self.testIter


    def classify(self, image_tensor):
        """
        Classifies input images using the trained model.

        Parameters:
        - image_tensor: Input images as a tensor.

        Returns:
        - predictions: Binary predictions for the input images.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient calculation for efficiency
        with torch.no_grad():      
            # Get model output
            model_output = self.model(image_tensor)['out']
            # Apply sigmoid activation to convert logits to probabilities
            preds = torch.sigmoid(model_output)

        # Convert probabilities to binary predictions
        predictions = (preds > 0.5).float()

        return predictions