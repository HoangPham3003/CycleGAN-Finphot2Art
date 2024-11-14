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
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand1(x11)
        x13 = self.expand2(x12)
        xn = self.downfeature(x13)
        return self.tanh(xn)
        
        
        