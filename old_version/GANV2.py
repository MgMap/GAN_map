import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
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

# Define the Generator
class Generator(nn.Module):
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
def train_gan(epochs, batch_size):
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

            # Calculate accuracy
            output_real_acc = output_real >= 0.5
            output_fake_acc = output_fake < 0.5
            acc_real = (output_real_acc.sum().float() / batch_size) * 100
            acc_fake = (output_fake_acc.sum().float() / batch_size) * 100

            # Print the progress
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(D_loss=(loss_real + loss_fake).item(), G_loss=loss_G.item(), Acc_real=acc_real.item(), Acc_fake=acc_fake.item())

        # Save generated images
        save_images(epoch + 1)
        # Save the Generator model
        save_model(generator, epoch + 1)

# Function to save the model
def save_model(model, epoch):
    model_dir = os.path.expanduser('~/Documents/GAN/models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), f"{model_dir}/generator_epoch_{epoch}.pth")

# Function to save a single generated image
def save_images(epoch):
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():
        noise = torch.randn(1, 100, device=device)
        gen_img = generator(noise).cpu()
        gen_img = 0.5 * gen_img + 0.5  # Rescale 0-1

        fig, ax = plt.subplots()
        ax.imshow(gen_img[0].permute(1, 2, 0))
        ax.axis('off')

        save_dir = os.path.expanduser('~/Documents/GAN/dog_gan_images')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f"{save_dir}/dog_gan_image_{epoch}.png")
        plt.close()
    generator.train()  # Set the generator back to training mode

# Train the GAN
train_gan(epochs=2000, batch_size=128)
