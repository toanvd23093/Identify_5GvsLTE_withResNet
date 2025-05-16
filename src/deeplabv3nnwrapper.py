import numpy as np
import torch
import src.deeplabv3nn as torch_module # PyTorch model module

###################################################################
#                   Neural Network Model
###################################################################
def construct_model(network_backbone, filename: str=""):
    
    # Load and return the default deeplabv3nn. 

    model_obj = torch_module.trainer(network_backbone, False, filename)

    return model_obj


###################################################################
#                   Initialize Trainer
###################################################################
def setup_trainer(network_backbone,
                  file_name: str, 
                  save_chk_pt_freq: int=100, 
                  max_train_iter: int=500):
    """
    Instantiate the PyTorch model trainer object. 

    Parameters:
    - network_backbone: The backbone architecture for the network.
    - file_name: The file name to save the model checkpoint.
    - save_chk_pt_freq: Frequency of saving checkpoints during training.
    - max_train_iter: Maximum number of training iterations.
    Returns:
    - trainer: Trainer object for PyTorch model
    """    
    
    model_obj = torch_module.trainer(network_backbone, True, file_name, save_chk_pt_freq, max_train_iter)

    return model_obj
    
###################################################################
#                   Train One Iteration
###################################################################
def train_one_iteration(model_obj, images: np.ndarray, masks: np.ndarray):
    """
    Performs a training step with the given images and masks.

    Parameters:
    - model_obj: Model object
    - images: A batch of input images.
    - masks: Corresponding ground truth masks.
    """
    # Preprocess the data from MATLAB format to PyTorch tensors
    image_tensor, mask_tensor = process_data_from_matlab(images, masks, model_obj.device)
    
    # Call the train_step method from the superclass with the processed data
    model_obj.train_step(image_tensor, mask_tensor)

    return model_obj.loss_vector[-1]

###################################################################
#                   Validation
###################################################################
def validate(model_obj, images: np.ndarray, masks: np.ndarray):
    """
    Performs a validation step with the given images and masks.

    Parameters:
    - model_obj: Model object
    - images: A batch of input images.
    - masks: Corresponding ground truth masks.
    """
    # Preprocess the data from MATLAB format to PyTorch tensors
    image_tensor, mask_tensor = process_data_from_matlab(images, masks, model_obj.device)

    # Call the validate_step method from the superclass with the processed data
    model_obj.validate_step(image_tensor, mask_tensor)


    # Return the average validation loss
    return model_obj.val_accuracy[-1]

###################################################################
#                   Test
###################################################################
def test(model_obj, images: np.ndarray, masks: np.ndarray):
    """
    Performs a test step with the given images and masks.

    Parameters:
    - model_obj: Model object
    - images: A batch of input images.
    - masks: Corresponding ground truth masks.
    """
    # Preprocess the data from MATLAB format to PyTorch tensors
    image_tensor, mask_tensor = process_data_from_matlab(images, masks, model_obj.device)

    # Call the test_step method from the superclass with the processed data
    model_obj.test_step(image_tensor, mask_tensor)

###################################################################
#                   Prediction
###################################################################
def predict(model_obj, images: np.ndarray):
        """
        Classifies the given images and returns the predictions.

        Parameters:
        - model_obj: Model object with trained model
        - images: A batch of input images.

        Returns:
        - predictions: The predicted pixel mask in a format suitable for post processing in MATLAB.
        """
        # Preprocess the data from MATLAB format to PyTorch tensors
        # An empty mask tensor is created as it's not needed for classification
        image_tensor, mask_tensor = process_data_from_matlab(images, np.empty_like(images), model_obj.device)

        # Call the classify method from the superclass with the processed image tensor
        network_output = model_obj.classify(image_tensor)

        # Postprocess the network output to convert it back to MATLAB format
        predictions = process_data_to_matlab(network_output)
        
        return predictions

###################################################################
#                   Model Information
###################################################################
def info(backbone,num_classes:int):
    """
    Return the total number of learnables and layers. 

    Parameters:
    - network_backbone: The backbone architecture for the network.
    - num_classes: Number of classes
    Returns:
    - num_layers: number of layers
    - total_params: total number of parameters
    """
    [num_layers, total_params] = torch_module.analyze_model(backbone,num_classes)

    return [num_layers, total_params]
    
###################################################################
#                   Helpers
###################################################################
def process_data_from_matlab(images, masks, device):
    """
    Processes image and mask data from MATLAB format to torch tensors.

    Parameters:
    - images: An array of images in MATLAB format.
    - masks: An array of masks in MATLAB format.
    - device: The device (CPU or GPU) to which the tensors should be moved.

    Returns:
    - image_tensor: A PyTorch tensor of images, formatted for model input.
    - mask_tensor: A PyTorch tensor of masks, formatted for model input.
    """
    # Convert numpy arrays to torch tensors without copying
    image_tensor = torch.from_numpy(images)
    mask_tensor = torch.from_numpy(masks)

    # Move tensors to specified device
    image_tensor = image_tensor.to(device) # if device is GPU, data is copied
    mask_tensor = mask_tensor.to(device) # if device is GPU, data is copied

    # Permute dimensions from HWCN --> NCHW to match the network input. No data copying 
    image_tensor = image_tensor.permute((3, 2, 0, 1))
    mask_tensor = mask_tensor.permute((3, 2, 0, 1))

    image_tensor = image_tensor.contiguous() # C_CONTIGUOUS, data is copied
    mask_tensor = mask_tensor.contiguous() # C_CONTIGUOUS, data is copied
    
    return image_tensor, mask_tensor

def process_data_to_matlab(network_output):
    """
    Processes network output from torch tensors to a format suitable for post processing in MATLAB.

    Parameters:
    - network_output: The output tensor from the network.

    Returns:
    - predictions: A numpy array of predictions, formatted for MATLAB.
    """
    # Permute dimensions of the network output from NCHW --> HWCN 
    predictions = network_output.permute((2, 3, 1, 0))

    predictions = predictions.contiguous()
    
    # Move predictions to CPU (if necessary)
    predictions = predictions.cpu() # if device used is GPU, data is copied

    # Convert to a numpy array. No data copying
    predictions = predictions.numpy()
    
    return predictions