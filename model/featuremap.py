import torch.nn as nn


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=7,
            padding=3,
            padding_mode='reflect'
        )
        
    
    def forward(self, x)
        x = self.conv(x)
        return x