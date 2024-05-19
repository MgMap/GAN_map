import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_cuda_devices)

    # Iterate over each CUDA device and print its properties
    for i in range(num_cuda_devices):
        print("CUDA Device", i)
        print("Name:", torch.cuda.get_device_name(i))
        print("CUDA Cores:", torch.cuda.get_device_capability(i))
        print("Memory:", torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), "GB")
        print("=" * 20)
else:
    print("CUDA is not available.")
