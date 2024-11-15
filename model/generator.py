import torch.nn as nn

from .residual import ResidualBlock
from .contracting import ContractingBlock
from .expanding import ExpandingBlock
from .featuremap import FeatureMapBlock


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        
        self.upfeature = FeatureMapBlock(input_channels=input_channels, output_channels=hidden_channels)
        self.contract1 = ContractingBlock(input_channels=hidden_channels)
        self.contract2 = ContractingBlock(input_channels=hidden_channels * 2)
        res_mult = 4
        self.res0 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res1 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res2 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res3 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res4 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res5 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res6 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res7 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.res8 = ResidualBlock(input_channels=hidden_channels * res_mult)
        self.expand1 = ExpandingBlock(input_channels=hidden_channels * 4)
        self.expand2 = ExpandingBlock(input_channels=hidden_channels * 2)
        self.downfeature = FeatureMapBlock(input_channels=hidden_channels, output_channels=output_channels)
        self.tanh = nn.Tanh()
        
    
    def forward(self, x):
        x = self.upfeature(x)
        x = self.contract1(x)
        x = self.contract2(x)
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.expand1(x)
        x = self.expand2(x)
        x = self.downfeature(x)
        return self.tanh(x)
        
        
        