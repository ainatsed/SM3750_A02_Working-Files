# Modified from L17.5 A Variational AutoEncoder for Handwritten Digits in PyTorch 
# https://youtu.be/afNuE5z2CQ8?si=dWmGaZJ3uIWA-bPx


import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from image_dataloader_2 import ImageDataset # import custom dataset

import numpy as np
import matplotlib.pyplot as plt

import os

path_dir = 'output'

    
'''
Model
'''
class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :28, :28]
    
    
class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # 2d conv layers
        self.encoder = nn.Sequential(
            # 1 input channel (image), output 32 conv feature vectors 
            # with a square kernel size of 3, padding of 1
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1), 
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.Flatten(),
        )
        
        # fully connected layers
        # output size, latent space
        self.z_mean = torch.nn.Linear(256, 2) 
        self.z_log_var = torch.nn.Linear(256, 2)
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(2, 256), 
            Reshape(-1, 64, 16, 16),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(), # trim last pixel
            nn.Sigmoid() # pixels are in 0-1 range
        )
        
    ''' REDUNDANT 
    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded  # update mean, s.d
    '''
    
    def reparameterize(self, z_mu, z_log_var):
        # sampled latent vector from normal distribution(mu, s.d)
        epsilon = torch.randn_like(z_log_var).to(DEVICE)
        #epsilon = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device()) 
        z = z_mu + z_log_var*epsilon  
        return z 
    
    def forward(self, x):
        x = self.encoder(x) # encode image into 2d
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var) # encoder produces mean and log of variance 
        decoded = self.decoder(encoded) # (i.e., parameters of simple tractable normal distribution "q"                 
        return encoded, decoded, z_mean, z_log_var  


def loss_function(encoded, decoded, mean, log_var):
    #reproduction_loss = nn.functional.binary_cross_entropy(decoded, encoded, reduction='sum')
    reproduction_loss = nn.functional.mse_loss(encoded, decoded, reduction='sum')
    
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def train_vae(epochs, model, optimizer, device,
              train_loader, loss_function=None,
              logging_interval=2,
              skip_epoch_stats=False,
              reconstruction_term_weight=1,
              save_model=None):
    
    log_dict = {'train_combined_loss_per_batch':[],
                'train_combined_loss_per_epoch':[],
                'train_reconstuction_loss_per_batch':[],
                'train_kl_loss_per_batch':[]}
    
    #if loss_fn is None:
        #loss_fn = F.mse_loss # use mean sq loss
        
    #start_time = time.time()
    
    for epoch in range(epochs):
        
        overall_loss = 0
        outputs = [] #array to store iterations
        
        for batch_idx, (x, _) in enumerate(loader):
            x = x.view(x.shape[0], 64, 64)
            x = x.to(DEVICE)
            
            # run forward and back propagation
            encoded, decoded, z_mean, z_log_var  = model(x)
            loss = loss_function(encoded, decoded, z_mean, z_log_var)
            overall_loss += loss.item()
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
           
        outputs.append((epoch, encoded, decoded)) #store epoch, real image, reconstructed
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
                
    

if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" 
                          if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available()  
                          else "cpu")
    print("device:", DEVICE)
       
    size = (64, 64) 
    
    #x_dim is H * W because the images are grayscale
    #rather than RGB. 
    #If we were dealing with RGB, x_dim should be multiplied by 3 for the
    #color channels. 
    x_dim  = size[0] * size[1]
        
    
    '''
    Instantiate the model 
    '''
    batch_size = 2
    data_path = 'images'
    size = (64, 64)
    learning_rate = 1e-3
    epochs = 50
    
    model = VAE()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
       
    '''
    Dataset processing
    '''
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor( ), 
                                                torchvision.transforms.Resize(size , antialias=True) ] ) 
                                               
    dataset = ImageDataset(data_path, transform=transform)
    
    # Split dataset into training and testing data
    train_set, test_set = torch.utils.data.random_split(dataset, [42, 12]) # dataset contains 54 images
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    print("Train Loader:", len(train_loader)) # 21    
    print("Test Loader:", len(test_loader)) # 6
    
    loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_size=batch_size, 
        # If true, shuffles the dataset at every epoch 
        shuffle=True
    ) 
     
    print("--------------Checking Dataset--------------")
    #testing that the length of the dataloader is correct. 
    print("The length of the dataloader is", len(loader))
     
     
    for batch_idx, (image) in enumerate(loader):
        print(image.shape)

    
    
    '''
    Training
    
    ''' 
    print("--------------Start training VAE--------------")
    log_dict = train_vae(epochs= epochs, model=model, 
                        optimizer=optimizer, device=DEVICE,
                        loss_function=loss_function,
                        logging_interval=2,
                        train_loader=train_loader,
                        skip_epoch_stats=True,
                        )
       
    print("--------------Finished--------------")
    
    print("--------------Plotting Results--------------") 
 
    with torch.no_grad():
        
        #generate a batch of fake images 
        noise = torch.randn(batch_size, 2).to(DEVICE)
        generated_images = model.Decoder(noise)
        
        
        #show two generated images
        for idx in range(2): 
            #assume grayscale image... 
            x = generated_images.view( batch_size, size[0], size[1])[idx].cpu()
            
            x = x.squeeze()
            
            x = x.numpy() 
            
            #change the range from (0,1) to (0,255)
            x = (x * 255)
            #convert to int datatype 
            x = x.astype(np.uint8)
            
            plt.gray()
            plt.figure()
            plt.imshow(x, interpolation='bicubic') 
        
 
