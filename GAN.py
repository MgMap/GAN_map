import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar
import numpy as np
import matplotlib.pyplot as plt
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your dataset
data_path = os.path.expanduser('~/Documents/GAN/stanford_dogs/Images')

# Image parameters
img_height, img_width = 64, 64
batch_size = 16  # Reduced batch size to lower memory usage

# Data augmentation and normalization for training
data_transforms = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = datasets.ImageFolder(root=data_path, transform=data_transforms)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    # Define the Generator
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, img_height * img_width * 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 3, img_height, img_width)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    # Define the Discriminator
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_height * img_width * 3, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Initialize Generator and Discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training function
def train_gan(epochs, batch_size, save_interval):
    for epoch in range(epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for i, data in loop:
            real_images, _ = data
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            label_real = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
            label_fake = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

            # Train Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images.view(-1, img_height * img_width * 3))
            loss_real = criterion(output_real, label_real)
            loss_real.backward()

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach().view(-1, img_height * img_width * 3))
            loss_fake = criterion(output_fake, label_fake)
            loss_fake.backward()

            optimizer_D.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images.view(-1, img_height * img_width * 3))
            loss_G = criterion(output, label_real)
            loss_G.backward()
            optimizer_G.step()

            # Print the progress
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(D_loss=(loss_real + loss_fake).item(), G_loss=loss_G.item())

            # Save generated images
            total_batches = len(train_loader) * epoch + i + 1
            if total_batches % save_interval == 0:
                save_images(total_batches)

# Function to save generated images
def save_images(epoch):
    with torch.no_grad():
        noise = torch.randn(25, 100, device=device)
        gen_imgs = generator(noise).cpu()
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale 0-1

        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_imgs[cnt].permute(1, 2, 0))
                axs[i, j].axis('off')
                cnt += 1
        save_dir = os.path.expanduser('~/Documents/GAN/gan_images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/gan_image_{epoch}.png")
        plt.close()

# Train the GAN
train_gan(epochs=20, batch_size=16, save_interval=1)
