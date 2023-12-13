# Modified from L17.5 A Variational AutoEncoder for Handwritten Digits in PyTorch
# https://youtu.be/ZoZHd0Zm3RY?si=zLAfpBzFIFt8XWbn

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

output_dir = 'output'

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       # (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
   
    
def loss_function(x, x_hat, mean, log_var):
    #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
    
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

    
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
    
    hidden_dim = 256
    latent_dim = 2
    
    
    
    
    '''
    Instantiate the model 
    '''
    
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
    
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    
    
    '''
    Dataset processing
    '''
    batch_size = 2
    data_path = 'images'
    size = (64, 64)
    learning_rate = 1e-3
    epochs = 200
    
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor( ), 
                                                torchvision.transforms.Resize(size , antialias=True) ] ) 
                                               
    dataset = ImageDataset(data_path, transform=transform)
    
    loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_size=batch_size, 
        # If true, shuffles the dataset at every epoch 
        shuffle=True
    ) 
     
    print("--------------Checking Dataset...--------------")
    #testing that the length of the dataloader is correct. 
    print("The length of the dataloader is", len(loader))
     
     
    for batch_idx, (image) in enumerate(loader):
        print(image.shape)
    
    
    '''
    Optimizer
    '''
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    
    '''
    Training
    
    '''
    
    print("--------------Start training VAE--------------")
    model.train()
    
    for epoch in range(epochs):
        
        overall_loss = 0
        outputs = [] #array to store iterations
        
        for batch_idx, (x, _) in enumerate(loader):
            
            x = x.view(x.shape[0], x_dim)
            
            x = x.to(DEVICE)
    
            optimizer.zero_grad()
    
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
           
        outputs.append((epoch, x, x_hat)) #store epoch, real image, reconstructed
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
        
        
    print("--------------Finished--------------")   
    
    
    #we finished training so we set model to eval( )
    model.eval()
                
    print("--------------Plotting Results--------------") 
 
    with torch.no_grad():
        
        #generate a batch of fake images 
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = model.Decoder(noise)
        
        
        #show two generated images
        for idx in range(2): 
            #assume grayscale image... 
            x = generated_images.view( batch_size, size[0], size[1])[idx].cpu()
            
            # save generated images
            #filepath = os.path.join(output_dir, "gen_image-{idx}.png")
            #torchvision.utils.save_image(x, filepath)
            
            x = x.squeeze()
            
            x = x.numpy() 
            
            #change the range from (0,1) to (0,255)
            x = (x * 255)
            #convert to int datatype 
            x = x.astype(np.uint8)
            
            plt.gray()
            plt.figure()
            plt.imshow(x, interpolation='bicubic') 
            
            

        
        










