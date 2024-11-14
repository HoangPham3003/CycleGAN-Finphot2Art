import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        
        self.upfeature = FeatureMapBlock(input_channels=in_channels, output_channels=hidden_channels)
        self.contract1 = ContractingBlock(
            input_channels=hidden_channels,
            use_bn=False,
            kernel_size=4,
            activation='lrelu'
        )
        self.contract2 = ContractingBlock(
            input_channels=hidden_channels * 2.
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
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.finald(x3)
        return xn
        