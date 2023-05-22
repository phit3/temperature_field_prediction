import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random

class TempPredNet(nn.Module):
    # define the loss function
    @staticmethod
    def loss_function(targets, outputs):
        loss = F.mse_loss(outputs, targets)
        return loss

    # initialize the model
    def __init__(self, size_factor: int = 48, seed: int = 0):
        super().__init__()
        self.size_factor = size_factor
        self.seed = seed
        self.relu_like = nn.ReLU()
        self._setting_seeds()
        self._build_model()
        
    # set seeds for reproducibility
    def _setting_seeds(self):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    # initialize layers of the model
    def _build_model(self):
            channel = [1, 2, 4, 8, 16, 32] * self.size_factor
            us_factor = 2
            self.pool = nn.MaxPool2d(us_factor, stride=us_factor, ceil_mode=True)
            
            self.conv1a = nn.Conv2d(3, channel[0], 3, stride=1, padding='same')
            self.bn1a = nn.BatchNorm2d(channel[0])
            self.conv1b = nn.Conv2d(channel[0], channel[0], 3, stride=1, padding='same')
            self.bn1b = nn.BatchNorm2d(channel[0])
            
            self.conv2a = nn.Conv2d(channel[0], channel[1], 3, stride=1, padding='same')
            self.bn2a = nn.BatchNorm2d(channel[1])
            self.conv2b = nn.Conv2d(channel[1], channel[1], 3, stride=1, padding='same')
            self.bn2b = nn.BatchNorm2d(channel[1])
            
            self.conv3a = nn.Conv2d(channel[1], channel[2], 3, stride=1, padding='same')
            self.bn3a = nn.BatchNorm2d(channel[2])
            self.conv3b = nn.Conv2d(channel[2], channel[2], 3, stride=1, padding='same')
            self.bn3b = nn.BatchNorm2d(channel[2])
            
            self.conv4a = nn.Conv2d(channel[2], channel[3], 3, stride=1, padding='same')
            self.bn4a = nn.BatchNorm2d(channel[3])
            self.conv4b = nn.Conv2d(channel[3], channel[3], 3, stride=1, padding='same')
            self.bn4b = nn.BatchNorm2d(channel[3])
            
            self.conv5 = nn.Conv2d(channel[3], channel[4], 3, stride=1, padding='same')
            self.bn5 = nn.BatchNorm2d(channel[4])
            
            self.upsample = nn.Upsample(scale_factor=2)

            self.dconv1 = nn.Conv2d(channel[4], channel[4], 3, stride=1, padding='same')
            self.dbn1 = nn.BatchNorm2d(channel[4])
            
            self.uconv1 = nn.Conv2d(channel[4], channel[3], 2, stride=1, padding='same')
            self.ubn1 = nn.BatchNorm2d(channel[3])
            
            self.dconv2a = nn.Conv2d(channel[4], channel[3], 3, stride=1, padding='same')
            self.dbn2a = nn.BatchNorm2d(channel[3])
            self.dconv2b = nn.Conv2d(channel[3], channel[3], 3, stride=1, padding='same')
            self.dbn2b = nn.BatchNorm2d(channel[3])
            
            self.uconv2 = nn.Conv2d(channel[3], channel[2], 2, stride=1, padding='same')
            self.ubn2 = nn.BatchNorm2d(channel[2])
            
            self.dconv3a = nn.Conv2d(channel[3], channel[2], 3, stride=1, padding='same')
            self.dbn3a = nn.BatchNorm2d(channel[2])
            self.dconv3b = nn.Conv2d(channel[2], channel[2], 3, stride=1, padding='same')
            self.dbn3b = nn.BatchNorm2d(channel[2])
            
            self.uconv3 = nn.Conv2d(channel[2], channel[1], 2, stride=1, padding='same')
            self.ubn3 = nn.BatchNorm2d(channel[1])
            
            self.dconv4a = nn.Conv2d(channel[2], channel[1], 3, stride=1, padding='same')
            self.dbn4a = nn.BatchNorm2d(channel[1])
            self.dconv4b = nn.Conv2d(channel[1], channel[1], 3, stride=1, padding='same')
            self.dbn4b = nn.BatchNorm2d(channel[1])
            
            self.uconv4 = nn.Conv2d(channel[1], channel[0], 2, stride=1, padding='same')
            self.ubn4 = nn.BatchNorm2d(channel[0])
            
            self.dconv5a = nn.Conv2d(channel[1], channel[0], 3, stride=1, padding='same')
            self.dbn5a = nn.BatchNorm2d(channel[0])
            self.dconv5b = nn.Conv2d(channel[0], channel[0], 3, stride=1, padding='same')
            self.dbn5b = nn.BatchNorm2d(channel[0])
            self.dconv5c = nn.Conv2d(channel[0], 1, 1, stride=1, padding='same')

    # define forward pass of the model
    def forward(self, img: torch.Tensor):
        # encoder
        x1 = self.conv1a(img)
        x1 = self.relu_like(x1)
        x1 = self.bn1a(x1)
        x1 = self.conv1b(x1)
        x1 = self.relu_like(x1)
        x1 = self.bn1b(x1)
        
        x2 = self.pool(x1)
        
        x2 = self.conv2a(x2)
        x2 = self.relu_like(x2)
        x2 = self.bn2a(x2)
        x2 = self.conv2b(x2)
        x2 = self.relu_like(x2)
        x2 = self.bn2b(x2)
        
        x3 = self.pool(x2)
        
        x3 = self.conv3a(x3)
        x3 = self.relu_like(x3)
        x3 = self.bn3a(x3)
        x3 = self.conv3b(x3)
        x3 = self.relu_like(x3)
        x3 = self.bn3b(x3)
        
        x4 = self.pool(x3)
        
        x4 = self.conv4a(x4)
        x4 = self.relu_like(x4)
        x4 = self.bn4a(x4)
        x4 = self.conv4b(x4)
        x4 = self.relu_like(x4)
        x4 = self.bn4b(x4)
        
        x5 = self.pool(x4)
        
        x5 = self.conv5(x5)
        x5 = self.relu_like(x5)
        x5 = self.bn5(x5)
        
        y1 = self.dconv1(x5)
        y1 = self.relu_like(y1)
        y1 = self.dbn1(y1)

        # decoder
        y2 = self.upsample(y1)
        y2 = self.uconv1(y2)
        y2 = self.ubn1(y2)

        y2 = torch.cat([x4, y2[:, :, :x4.shape[2], :x4.shape[3]]], dim=1)
        
        y2 = self.dconv2a(y2)
        y2 = self.relu_like(y2)
        y2 = self.dbn2a(y2)
        y2 = self.dconv2b(y2)
        y2 = self.relu_like(y2)
        y2 = self.dbn2b(y2)
        
        y3 = self.upsample(y2)
        y3 = self.uconv2(y3)
        y3 = self.ubn2(y3)

        y3 = torch.cat([x3, y3[:, :, :x3.shape[2], :x3.shape[3]]], dim=1)
        
        y3 = self.dconv3a(y3)
        y3 = self.relu_like(y3)
        y3 = self.dbn3a(y3)
        y3 = self.dconv3b(y3)
        y3 = self.relu_like(y3)
        y3 = self.dbn3b(y3)
        
        y4 = self.upsample(y3)
        y4 = self.uconv3(y4)
        y4 = self.ubn3(y4)

        y4 = torch.cat([x2, y4[:, :, :x2.shape[2], :x2.shape[3]]], dim=1)
        
        y4 = self.dconv4a(y4)
        y4 = self.relu_like(y4)
        y4 = self.dbn4a(y4)
        y4 = self.dconv4b(y4)
        y4 = self.relu_like(y4)
        y4 = self.dbn4b(y4)
        
        y5 = self.upsample(y4)
        y5 = self.uconv4(y5)
        y5 = self.ubn4(y5)

        y5 = torch.cat([x1, y5[:, :, :x1.shape[2], :x1.shape[3]]], dim=1)
        
        y5 = self.dconv5a(y5)
        y5 = self.relu_like(y5)
        y5 = self.dbn5a(y5)
        y5 = self.dconv5b(y5)
        y5 = self.relu_like(y5)
        y5 = self.dbn5b(y5)
        y5 = self.dconv5c(y5)

        y5 = torch.sigmoid(y5)[:, :, :img.shape[2], :img.shape[3]]
        
        return y5
