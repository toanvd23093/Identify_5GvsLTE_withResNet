import src.deeplabv3nnwrapper as dlv3nnwrapper
import os

# Test
def test_train_one_iteration_perf(loadMATLABData,benchmark):

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
    benchmark.pedantic(dlv3nnwrapper.train_one_iteration, args=(model_obj, train_images, train_masks,), iterations=1,rounds=1)
    

