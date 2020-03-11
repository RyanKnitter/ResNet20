import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=2)
        self.norm1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(3, stride=1)    
        
        
        self.conv16_1 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm16_1 = nn.BatchNorm2d(16)
        self.conv16_2 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm16_2 = nn.BatchNorm2d(16)
        
        self.conv16_3 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm16_3 = nn.BatchNorm2d(16)
        self.conv16_4 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm16_4 = nn.BatchNorm2d(16)
        
        self.conv16_5 = nn.Conv2d(16, 16, 3, padding=1)
        self.norm16_5 = nn.BatchNorm2d(16)
        self.conv16_6 = nn.Conv2d(16, 32, 3, padding=1)
        self.norm16_6 = nn.BatchNorm2d(32)
        
        
        self.conv32_1 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.norm32_1 = nn.BatchNorm2d(32)
        self.conv32_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm32_2 = nn.BatchNorm2d(32)
        
        self.conv32_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.norm32_3 = nn.BatchNorm2d(32)
        self.conv32_4 = nn.Conv2d(32, 32, 3, padding=1)
        self.norm32_4 = nn.BatchNorm2d(32)
        
        self.conv32_5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.norm32_5 = nn.BatchNorm2d(32)
        self.conv32_6 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm32_6 = nn.BatchNorm2d(64)
        
        
        self.conv64_1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.norm64_1 = nn.BatchNorm2d(64)
        self.conv64_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm64_2 = nn.BatchNorm2d(64)
        
        self.conv64_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.norm64_3 = nn.BatchNorm2d(64)
        self.conv64_4 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm64_4 = nn.BatchNorm2d(64)
        
        self.conv64_5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.norm64_5 = nn.BatchNorm2d(64)
        self.conv64_6 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm64_6 = nn.BatchNorm2d(64)
        
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(64,10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        
        identity = x
        x = F.relu(self.norm16_1(self.conv16_1(x)))
        x = self.norm16_2(self.conv16_2(x))
        x += identity
        x = F.relu(x)
        
        identity = x
        x = F.relu(self.norm16_3(self.conv16_3(x)))
        x = self.norm16_4(self.conv16_4(x))
        x += identity
        x = F.relu(x)
        
        identity = F.pad(x,(0,0,0,0,0,16))
        x = F.relu(self.norm16_5(self.conv16_5(x)))
        x = self.norm16_6(self.conv16_6(x))
        x += identity
        x = F.relu(x)
        

        identity = F.avg_pool2d(identity,1,2)
        x = F.relu(self.norm32_1(self.conv32_1(x)))
        x = self.norm32_2(self.conv32_2(x))
        x += identity
        x = F.relu(x)
        
        identity = x
        x = F.relu(self.norm32_3(self.conv32_3(x)))
        x = self.norm32_4(self.conv32_4(x))
        x += identity
        x = F.relu(x)
        
        identity = F.pad(x,(0,0,0,0,0,32))
        x = F.relu(self.norm32_5(self.conv32_5(x)))
        x = self.norm32_6(self.conv32_6(x))
        x += identity
        x = F.relu(x)
        
        
        identity = F.avg_pool2d(x,1,2)
        x = F.relu(self.norm64_1(self.conv64_1(x)))
        x = self.norm64_2(self.conv64_2(x))
        x += identity
        x = F.relu(x)
        
        identity = x
        x = F.relu(self.norm64_3(self.conv64_3(x)))
        x = self.norm64_4(self.conv64_4(x))
        x += identity
        x = F.relu(x)
        
        identity = x
        x = F.relu(self.norm64_5(self.conv64_5(x)))
        x = self.norm64_6(self.conv64_6(x))
        x += identity
        x = F.relu(x)
        
        
        x = self.pool2(x)
        x = torch.flatten(x, 1)        
        x = self.fc(x)
        
        return x
