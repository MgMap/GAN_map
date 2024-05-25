import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from utils import save_model, save_images, train_gan
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your dataset
data_path = os.path.expanduser('~/Documents/GAN/stanford_dogs/Images')

# Image parameters
img_height, img_width = 64, 64
batch_size = 128

# Data augmentation and normalization for training
data_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(root=data_path, transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize Generator and Discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define loss function and optimizers
criterion = torch.nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Train the GAN
train_gan(generator, discriminator, criterion, optimizer_G, optimizer_D, train_loader, device, epochs=2000, start_epoch=0)
