import torch.nn as nn

class ContractingBlock(nn.Module):
    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=input_channels * 2
            kernel_size=kernel_size,
            padding=1,
            stride=2,
            padding_mode='reflect'
        )
        
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(negative_slope=0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(num_features=input_channels * 2)
        self.use_bn = use_bn
        
    
    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x
        