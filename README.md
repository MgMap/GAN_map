# GAN (Generative adversarial network)

# Create a directory for the dataset
mkdir -p stanford_dogs

cd stanford_dogs

# Download the dataset
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

wget http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar

# Extract the images
tar -xvf images.tar

tar -xvf annotation.tar

# Read Me

This GAN model use GPU which save the RAM usage and train data faster than CPU training.

Using RAM makes not only slowdown PC, also crash and makes memory insufficient while in training process.

Just run the python file and it will generate the image. Have Fun.

# Final Result

![gan_image_25740](https://github.com/BarbatosV2/GAN/assets/63419320/76da3070-8d35-4cf4-b2e9-300c030acfe6)

Looks like dogs to me XD

