import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_model(generator, discriminator, optimizer_G, optimizer_D, epoch):
    model_dir = os.path.expanduser('~/Documents/GAN/models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, f"{model_dir}/generator_epoch_{epoch}.pth")

def save_images(generator, epoch, device):
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

def train_gan(generator, discriminator, criterion, optimizer_G, optimizer_D, train_loader, device, epochs, start_epoch=0):
    for epoch in range(start_epoch, epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for i, data in loop:
            real_images, _ = data
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            label_real = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
            label_fake = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

            # Train Discriminator
            discriminator.zero_grad()
            output_real = discriminator(real_images.view(-1, 64 * 64 * 3))
            loss_real = criterion(output_real, label_real)
            loss_real.backward()

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach().view(-1, 64 * 64 * 3))
            loss_fake = criterion(output_fake, label_fake)
            loss_fake.backward()

            optimizer_D.step()

            # Train Generator
            generator.zero_grad()
            output = discriminator(fake_images.view(-1, 64 * 64 * 3))
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
        save_images(generator, epoch + 1, device)
        # Save the Generator model
        save_model(generator, discriminator, optimizer_G, optimizer_D, epoch + 1)
