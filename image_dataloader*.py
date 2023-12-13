# Modified from week_12c_dataloader

import os 
import pandas as pd
import numpy as np 
from PIL import Image 
from skimage import io

import matplotlib.pyplot as plt 
  
import torch 
import torchvision
from torch.utils.data import Dataset


#custom dataset class 
class ImageDataset(Dataset): 
    
    def __init__(self, csv_file, image_dir, transform=None): 
        self.annotations = pd.read_csv(csv_file) 
        self.image_dir = image_dir
        self.transform = transform 
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.annotations) #54
  
    # Defining the procedure to obtain one item from the dataset 
    def __getitem__(self, index): 
        image_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0]) #row i, col 0: file names
        
        image = io.imread(image_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1])) #row i, col 1: class
        
    
        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
        
        #print("Image after transform", image.shape) #chanels, H< W
        return (image, y_label) # image, label


# Checking the dataset
if __name__ == "__main__" : 
    
    batch_size = 2
    size = (64, 64)
    
    # transformer
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                torchvision.transforms.Resize(size, antialias=True)]) 
    # dataloader                                
    dataset = ImageDataset(csv_file = 'fragments.csv', image_dir = 'images', transform=transform)
    print(len(dataset), "entries found in dataset.");
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_size=batch_size, 
        # If true, shuffles the dataset at every epoch 
        shuffle=True
    ) 
    
    #testing that the length of the dataloader is correct. 
    print("The length of the dataloader is", len(dataloader))
     
     
    for batch_idx, (image, y_label) in enumerate(dataloader):
        print(batch_idx, image.shape)
        
    print("--------------Finished--------------")    
