import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )    
        
        self.conv2 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            padding=1,
            padding_mode='reflect'
        )
        
        self.instancenorm = nn.InstanceNorm2d(num_features=input_channels)
        self.activation = nn.ReLU()
        
    
    def forward(self, x):
        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x