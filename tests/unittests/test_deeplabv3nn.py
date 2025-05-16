import torch
import src.deeplabv3nn as dlv3nn

# Test Analyze Model Method
def test_analyzeModel_resnet50():
    num_layers, total_params = dlv3nn.analyze_model(backbone="resnet50",num_classes=3)
    
    assert num_layers == 189, "Number of layers of ResNet50 should be 189"
    assert total_params == 41994822, "Total Params of ResNet50 should be 41994822"

# Test Analyze Model Method
def test_analyzeModel_resnet101():
    num_layers, total_params = dlv3nn.analyze_model(backbone="resnet101",num_classes=3)
    
    assert num_layers == 325, "Number of layers of ResNet101 should be 325"
    assert total_params == 60986950, "Total Params of ResNet101 should be 60986950"

# Test Dice BCE Loss Method
def test_diceBCELoss_forward(mockDataforLossFunction):
    lossFunc = dlv3nn.diceBCELoss()   
    image_tensor,mask_tensor = mockDataforLossFunction
    Dice_BCE_Act = lossFunc.forward(image_tensor, mask_tensor, smooth=1)
    Dice_BCE_Exp = torch.tensor(2.3133, dtype=torch.float64)

    assert torch.allclose(Dice_BCE_Act, Dice_BCE_Exp, atol=1e-4), "Output from diceBCELoss function is not correct"

# Test IOU Loss Method
def test_IOULoss_forward(mockDataforLossFunction):
    lossFunc = dlv3nn.IOU()
    image_tensor,mask_tensor = mockDataforLossFunction
    IOU_Act = lossFunc.forward(image_tensor, mask_tensor, smooth=1)
    IOU_Exp = torch.tensor(1.7393e-6, dtype=torch.float64)

    assert torch.allclose(IOU_Act, IOU_Exp, atol=1e-4), "Output from diceBCELoss function is not correct"