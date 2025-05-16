import pytest
from scipy.io import loadmat
import os
import numpy as np
import torch

@pytest.fixture
def mockDataforLossFunction():
    # Mocking the inputs and targets numpy arrays
    inputs = np.array(np.ones([256,256,3,4]))
    targets = np.array(np.zeros([256,256,3,4]))

    # Convert to tensors
    image_tensor = torch.from_numpy(inputs)
    mask_tensor = torch.from_numpy(targets)

    # Permute dimensions from HWCN --> NCHW to match the network input
    image_tensor = image_tensor.permute((3, 2, 0, 1))
    mask_tensor = mask_tensor.permute((3, 2, 0, 1))

    image_tensor = image_tensor.contiguous() # C_CONTIGUOUS, data is copied
    mask_tensor = mask_tensor.contiguous() # C_CONTIGUOUS, data is copied

    return [image_tensor,mask_tensor]

@pytest.fixture
def loadMATLABData():
    # Loading samples input from MATLAB
    # Train images and labels
    train_images_file_path = os.path.join(os.path.dirname(__file__), "mockData", "images_train.mat")
    train_masks_file_path  = os.path.join(os.path.dirname(__file__), "mockData", "masks_train.mat")
    
    train_images_data = loadmat(train_images_file_path)
    train_masks_data = loadmat(train_masks_file_path)

    train_images = train_images_data['images']
    train_masks = train_masks_data['masks']

    # Validation images and labels
    val_images_file_path = os.path.join(os.path.dirname(__file__), "mockData", "images_validation.mat")
    val_masks_file_path  = os.path.join(os.path.dirname(__file__), "mockData", "masks_validation.mat")

    val_images_data = loadmat(val_images_file_path)
    val_masks_data = loadmat(val_masks_file_path)
    
    val_images = val_images_data['images']
    val_masks = val_masks_data['masks']

    # Test images and labels
    test_images_file_path = os.path.join(os.path.dirname(__file__), "mockData", "images_test.mat")
    test_masks_file_path  = os.path.join(os.path.dirname(__file__), "mockData", "masks_test.mat")

    test_images_data = loadmat(test_images_file_path)
    test_masks_data = loadmat(test_masks_file_path)
    
    test_images = test_images_data['images']
    test_masks = test_masks_data['masks']

    return [train_images, train_masks, val_images, val_masks, test_images, test_masks]