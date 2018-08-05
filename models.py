## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# cause the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def try_gpu(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(2, 2), stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=1, padding=0)
        
        # Max-Pool layer that we will use multiple times
        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool = nn.MaxPool2d(2,2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.5)
        
        # Fully connected layers
        self.fc0 = nn.Linear(in_features=18432, out_features=6400)
        self.fc1 = nn.Linear(in_features=6400, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=136)
        #self.fc2 = nn.Linear(in_features=1000, out_features=500)
        #self.fc3 = nn.Linear(in_features=500, out_features=136)
        
        
        self.bn1 = nn.BatchNorm2d(num_features=32, eps=1e-05)
        self.bn2 = nn.BatchNorm2d(num_features=64, eps=1e-05)
        self.bn3 = nn.BatchNorm2d(num_features=128, eps=1e-05)
        self.bn4 = nn.BatchNorm2d(num_features=256, eps=1e-05)
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-05)
        self.bn6 = nn.BatchNorm1d(num_features=6400, eps=1e-05)
        self.bn7 = nn.BatchNorm1d(num_features=1000, eps=1e-05)
        
        


        # Custom weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv layers have weights initialized with random # drawn from uniform distribution
                m.weight = nn.init.xavier_uniform_(m.weight, gain=1)
            elif isinstance(m, nn.Linear):
                # FC layers have weights initialized with Glorot uniform initialization
                m.weight = nn.init.xavier_uniform_(m.weight, gain=1)
       
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        log = True;
        #if log:
         #  print('x: ' + str(x.shape))
        ## Conv layers

        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout1(x)
        #if log:
         #  print('conv1: ' + str(x.shape))
                
        x = self.pool(F.elu(self.conv2(x)))
        x= self.bn2(x)
        x = self.dropout2(x)
        #if log:
         #  print('conv2: ' + str(x.shape))
        
        x = self.pool(F.elu(self.conv3(x)))
        x= self.bn3(x)
        x = self.dropout3(x)
        #if log:
         #  print('conv3: ' + str(x.shape))
        
        x = self.pool(F.elu(self.conv4(x)))
        x= self.bn4(x)
        x = self.dropout4(x)
        
        x = self.pool(F.elu(self.conv5(x)))
        x= self.bn5(x)
        x = self.dropout5(x)
        #if log:
            #print('conv5: ' + str(x.shape))
        ## Flatten
        x = x.view(x.size(0), -1) # .view() can be thought as np.reshape
        
        #if log:
           #print('beforefc1: ' + str(x.shape))
        ## Fully connected layers
        x = F.elu(self.fc0(x))
        x= self.bn6(x)
        x = self.dropout6(x)
        
        x = F.elu(self.fc1(x))
        x= self.bn7(x)
        x = self.dropout7(x)
        
        #if log:
         #  print('afterfc1: ' + str(x.shape))
                
        x = self.fc2(x)
        #x = F.tanh(self.fc2(x)) # experimenting w/ different activation function instead of relu
        #x = self.dropout6(x)
                
        #x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 1, 5)
      
        self.fc0 = nn.Linear(in_features=49729, out_features=136)
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        log = True;
        if log:
           print('convbefore: ' + str(x.shape))
        x = F.elu(self.conv1(x))
        if log:
           print('convafter: ' + str(x.shape))
        x = x.view(x.size(0), -1) # .view() can be thought as np.reshape
        x = self.fc0(x)
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
