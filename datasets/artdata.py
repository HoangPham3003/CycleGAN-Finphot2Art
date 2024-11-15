import os
import glob
import random
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset


class ArtDataset(Dataset):
    def __init__(self, DATA_DIR='', 
                 transforms_=None,
                 unaligned=False,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        
        self.files_A = sorted(glob.glob(os.path.join(DATA_DIR, f'{mode}A') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(DATA_DIR, f'{mode}B') + '/*.*'))
        
    
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            
        return (item_A, item_B)
    
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
        