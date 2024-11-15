import torch.nn as nn

from .featuremap import FeatureMapBlock
from .contracting import ContractingBlock


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        
        self.upfeature = FeatureMapBlock(input_channels=input_channels, output_channels=hidden_channels)
        self.contract1 = ContractingBlock(
            input_channels=hidden_channels,
            use_bn=False,
            kernel_size=4,
            activation='lrelu'
        )
        self.contract2 = ContractingBlock(
            input_channels=hidden_channels * 2,
            kernel_size=4,
            activation='lrelu'
        )
        self.contract3 = ContractingBlock(
            input_channels=hidden_channels*4,
            kernel_size=4,
            activation='lrelu'
        )
        self.final = nn.Conv2d(
            in_channels=hidden_channels * 8, 
            out_channels=1, 
            kernel_size=1
        )
        
    
    def forward(self, x):
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.contract3(x)
        x = self.final(x)
        return x
        