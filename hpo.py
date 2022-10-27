#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    # set model to eval mode 
    model.eval()
    running_loss =0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data=data.to(device) # need to put data on GPU device
            target=target.to(device)
            pred = model(data)  # get the index of the max log-probability
            loss = criterion(pred, target)             # Caclulates loss
            running_loss+=loss
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item() # check output with actual label 
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} = {100*(correct/len(test_loader.dataset))}%)')
    test_loss = running_loss/len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}".format(
            test_loss
        )
    )
    

def train(model, train_loader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train() # set model to train mode 
    for e in range(epochs):
        running_loss=0
        correct=0
        for data, target in train_loader:         # Iterates through batches
            data=data.to(device) # need to put data on GPU device
            target = target.to(device)
            optimizer.zero_grad()                 # Reset gradients for new batch
            pred = model(data)                    # Runs forward pass
            loss = criterion(pred, target)             # Caclulates loss
            running_loss+=loss
            loss.backward()                       # Calculates gradients for all model parameters
            optimizer.step()                      # Updates weights 
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()  # Check how many correct predictions
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, Accuracy {100*(correct/len(train_loader.dataset))}%")
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    # add one fully connected layer 
    # number of classes we have (here: 133)
    model.fc = nn.Sequential(
               nn.Linear(num_features, 256),
               nn.ReLU(inplace=True),
               nn.Linear(256, 133))

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    return data_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    logger.info('Start main')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # need GPU to run

    cnn_model=net()
    cnn_model = cnn_model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    #SGD = stochastic gradient descent optimizer 
    optimizer = optim.SGD(cnn_model.parameters(), lr = args.lr, momentum = 0.9)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # Transform datasets
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    
    train_data = torchvision.datasets.ImageFolder(root=args.train, transform= transform)
    val_data = torchvision.datasets.ImageFolder(root=args.val, transform= transform)
    test_data = torchvision.datasets.ImageFolder(root=args.test, transform= transform)

    train_loader = create_data_loaders(train_data, args.batch_size)
    val_loader = create_data_loaders(val_data, args.batch_size)
    test_loader = create_data_loaders(test_data, args.batch_size)

    model=train(cnn_model, train_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "cnn_model_hpo.pth")
    torch.save(model, path)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(cnn_model, test_loader, loss_criterion, device)
    
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "-epochs",
        "--epochs",
        dest="epochs",
        type=int)
    
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        
    parser.add_argument(
        "--train", type=str, required=False, default=os.environ.get("SM_CHANNEL_TRAIN")
    )
    parser.add_argument(
        "--val", type=str, required=False, default=os.environ.get("SM_CHANNEL_VAL")
    )
    parser.add_argument(
        "--test", type=str, required=False, default=os.environ.get("SM_CHANNEL_TEST")
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=256
    )
    
    parser.add_argument(
        "--lr", type=float, required=False, default=0.001
    )
    
    args=parser.parse_args()
    
    main(args)
