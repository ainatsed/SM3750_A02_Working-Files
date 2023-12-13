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
class ImageDataset(torch.utils.data.Dataset): 
    
    def __init__(self, image_dir, RGB = False, transform=None): 
        self.data_dir = image_dir
        self.images = os.listdir(image_dir) 
        self.transform = transform 
        self.RGB = RGB
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.images) 
  
    # Defining the procedure to obtain one item from the dataset 
    def __getitem__(self, index): 
        image_path = os.path.join(self.data_dir, self.images[index]) 
        
        image = Image.open(image_path)
        
        if self.RGB: 
            image = image.convert("RGB")
        else: 
            image = image.convert("L")
        
        image = np.array(image) 
        
        #print("image from Pil", image.shape) #H, W, channels
  
        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
        
        #print("Image after transform", image.shape) #chanels, H< W
        return image


if __name__ == "__main__" : 
    
    batch_size = 2 # small dataset
    data_path = 'images'
    size = (64, 64)
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor( ), 
                                                torchvision.transforms.Resize(size , antialias=True) ] ) 
                                               
    dataset = ImageDataset(data_path, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_size=batch_size, 
        # If true, shuffles the dataset at every epoch 
        shuffle=True
    ) 
    
    #testing that the length of the dataloader is correct. 
    print("The length of the dataloader is", len(dataloader))
     
     
    for batch_idx, (image) in enumerate(dataloader):
        print(image.shape)
