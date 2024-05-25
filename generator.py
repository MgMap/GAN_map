import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image parameters
img_height, img_width = 64, 64

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

# Function to load the model
def load_model(model, epoch):
    model_dir = os.path.expanduser('~/Documents/GAN/models')
    model_path = f"{model_dir}/generator_epoch_{epoch}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to generate and save one image
def generate_images(model, num_images=1):
    with torch.no_grad():
        noise = torch.randn(num_images, 100, device=device)
        gen_imgs = model(noise).cpu()
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to 0-1

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(gen_imgs[0].permute(1, 2, 0))
        ax.axis('off')

        # Ensure the save directory exists
        save_dir = os.path.join(os.path.dirname(__file__), 'generatedImages')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Find the next available filename
        file_count = len([name for name in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, name))])
        file_name = f'generated_image_{file_count + 1}.png'

        # Save the figure
        fig.savefig(os.path.join(save_dir, file_name))
        plt.close()

# Load the trained model
generator_loaded = Generator().to(device)
generator_loaded = load_model(generator_loaded, epoch=2000)  # Adjust epoch number as needed

# Generate and save one image
generate_images(generator_loaded, num_images=1)
