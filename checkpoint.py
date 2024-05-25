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

# Find the last saved model file
model_dir = os.path.expanduser('~/Documents/GAN/models')
model_files = [f for f in os.listdir(model_dir) if f.startswith('generator_epoch_') and f.endswith('.pth')]
if model_files:
    latest_model_file = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    start_epoch = int(latest_model_file.split('_')[-1].split('.')[0])
    checkpoint = torch.load(os.path.join(model_dir, latest_model_file))
    try:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    except KeyError as e:
        print(f"KeyError: {e}. Make sure the checkpoint contains the correct keys.")
else:
    start_epoch = 0

# Train the GAN from the last saved checkpoint
train_gan(generator, discriminator, criterion, optimizer_G, optimizer_D, train_loader, device, epochs=2000, start_epoch=start_epoch + 1)
