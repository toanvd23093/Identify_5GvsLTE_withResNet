import src.deeplabv3nn as dlv3nn
import src.deeplabv3nnwrapper as dlv3nnwrapper
import os

# Test Construct Model Method
def test_construct_model():
    networkBackbone = 'resnet50'
    weightsFileName = os.path.join(os.path.dirname(__file__), "..", "mockData", "specsenselabv3.pth")
    model_obj = dlv3nnwrapper.construct_model(networkBackbone, weightsFileName)

    model_Act = model_obj.model
    model_Exp = dlv3nn.load_model(backbone=networkBackbone,num_classes=3)
    
    # Compare structure only
    assert str(model_Act) == str(model_Exp), "Model structures are different"

# Test Setup Trainer Method
def test_setup_trainer():
    networkBackbone = 'resnet50'
    weightsFileName = os.path.join(os.path.dirname(__file__), "..", "mockData", "defaultDeeplabv3.pth")
    saveCheckPointFreq = 50
    maxTrainIter = 500

    model_obj = dlv3nnwrapper.setup_trainer(networkBackbone,
                                            weightsFileName, 
                                            saveCheckPointFreq,
                                            maxTrainIter)

    model_Act = model_obj.model
    model_Exp = dlv3nn.load_model(backbone=networkBackbone,num_classes=3)
    
    # Compare structure only
    assert str(model_Act) == str(model_Exp), "Model structures are different"

# Test Info Method
def test_info():
    networkBackbone = 'resnet50'
    num_layers, total_params = dlv3nnwrapper.info(networkBackbone,3)

    assert num_layers == 189, "Number of layers of ResNet50 should be 189"
    assert total_params == 41994822, "Total Params of ResNet50 should be 41994822"

# Test Train/Validate/Test/Predict workflow
def test_train_one_iteration_val_test_predict(loadMATLABData):
    networkBackbone = 'resnet50'
    weightsFileName = os.path.join(os.path.dirname(__file__), "..", "mockData", "defaultDeeplabv3.pth")
    saveCheckPointFreq = 1
    maxTrainIter = 1

    model_obj = dlv3nnwrapper.setup_trainer(networkBackbone,
                                            weightsFileName, 
                                            saveCheckPointFreq,
                                            maxTrainIter)
    
    train_images, train_masks, val_images, val_masks, test_images, test_masks = loadMATLABData

    # Training for one iteration
    training_loss_Act = dlv3nnwrapper.train_one_iteration(model_obj, train_images, train_masks)
    assert training_loss_Act > 0, "Training loss for one iteration should be positive"

    # Validation
    val_accuracy_Act = dlv3nnwrapper.validate(model_obj, val_images, val_masks)
    assert val_accuracy_Act > 0, "Validation accuracy for one iteration should be positive"

    # Testing
    dlv3nnwrapper.test(model_obj, test_images, test_masks)
    test_iou_Act = model_obj.iou / model_obj.testIter
    test_accuracy_Act = model_obj.num_correct_test.item() / model_obj.num_pixels_test * 100
    test_dice_score_Act = model_obj.dice_score_test.item() / model_obj.testIter

    assert test_iou_Act > 0, "Test IOU should be positive"
    assert test_accuracy_Act > 0, "Test Accuracy should be positive"
    assert test_dice_score_Act > 0, "Test Dice Score should be positive"

    # Predict
    predictions = dlv3nnwrapper.predict(model_obj, test_images)
    assert predictions.shape == (256,256,3,6), "Predicted pixel structures are different"

