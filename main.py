import os
import re
import zipfile
import subprocess

from tqdm.auto import tqdm

from datasets.download import load_data
from datasets.dataloader import ArtLoader

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()
    
    DATA_DIR = load_data(ROOT_DIR)
    
    loader = ArtLoader(DATA_DIR=DATA_DIR)
    train_loader, test_loader = loader.load_data()
    
    print(len(train_loader))
    print(len(test_loader))
    
    
    
        
    