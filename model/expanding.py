import torch.nn as nn


class ExpandingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        
        self.conv1 = nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=input_channels // 2,
            kernel_size=3,
            stride=2, 
            padding=1,
            output_padding=1
        )
        
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(num_features=input_channels//2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x    
        