import tifffile as tiff
import numpy as np
import os
from tqdm import tqdm  # For progress tracking

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn.functional as F
import torch.fft
import matplotlib.pyplot as plt

# Step 1: Load the input image and PSF using tifffile
input_path = '/gpfsanonymousVCD_dataset/fixed_fish/240725_03/g/00000300.tif'
PSF_path = '/gpfsanonymousVCD_dataset/PSF_G.tif'

# Load the input image (assumed to be single-channel 2D: 2048x2048x1)
input_image = tiff.imread(input_path)

# Load the PSF image (assumed to be 3D: 2048x2048x300)
psf_stack = tiff.imread(PSF_path)

# Convert input image and PSF stack to float32 and normalize them
input_image = input_image.astype(np.float32) / np.max(input_image)
psf_stack = psf_stack.astype(np.float32) / np.max(psf_stack)

# Step 2: Move data to GPU using PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert the input image and PSF stack to PyTorch tensors
input_image_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
    device)  # Shape: (1, 1, 2048, 2048)
psf_stack_tensor = torch.tensor(psf_stack, dtype=torch.float32).permute(2, 0, 1).to(device)  # Shape: (300, 2048, 2048)

# Step 3: Apply padding (128 pixels) to the input image and PSF slices
input_image_padded = F.pad(input_image_tensor, (128, 128, 128, 128))  # Padding on each side
psf_stack_padded = F.pad(psf_stack_tensor, (128, 128, 128, 128))  # Padding for PSF slices


# Step 4: Convolve the padded input image with each PSF slice in a memory-efficient way
def apply_fft_convolution_by_layers(image_tensor, psf_stack_tensor):
    num_slices = psf_stack_tensor.shape[0]

    # Store results on CPU to save GPU memory
    blurred_images_cpu = []

    # Perform FFT on the input image
    image_fft = torch.fft.fft2(image_tensor)

    # Process each PSF slice independently
    for i in tqdm(range(num_slices), desc="Processing PSF Slices"):
        psf_slice = psf_stack_tensor[i, :, :].unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, 2176, 2176)

        # Perform FFT on PSF slice
        psf_fft = torch.fft.fft2(psf_slice,
                                 s=[image_tensor.shape[-2], image_tensor.shape[-1]])  # Match input image size

        # Multiply in the Fourier domain (point-wise multiplication)
        result_fft = image_fft * psf_fft

        # Perform inverse FFT to get the blurred image
        blurred_image = torch.fft.ifft2(result_fft).real  # Get real part of the IFFT result

        # Move the blurred image to CPU to save GPU memory
        blurred_images_cpu.append(blurred_image.cpu())

    return blurred_images_cpu


# Apply FFT convolution slice by slice
blurred_images_cpu = apply_fft_convolution_by_layers(input_image_padded, psf_stack_padded)


# Step 5: Estimate depth by comparing the input image with the blurred images in chunks
def estimate_depth_from_blur_in_chunks(blurred_images_cpu, original_image_tensor, chunk_size=10):
    num_slices = len(blurred_images_cpu)
    height, width = original_image_tensor.shape[-2:]

    # Initialize a placeholder for storing minimum differences and depth map
    min_diff = torch.full((height, width), float('inf'), device='cpu')  # Initialize with high values
    depth_map = torch.zeros((height, width), dtype=torch.long, device='cpu')  # Depth map on CPU

    # Process in chunks to avoid memory overflow
    for i in tqdm(range(0, num_slices, chunk_size), desc="Calculating depth map"):
        end = min(i + chunk_size, num_slices)
        chunk_blurred_images = blurred_images_cpu[i:end]  # Take a chunk of blurred images

        # Convert chunk to tensor and move to GPU
        chunk_blurred_images_tensor = torch.stack(chunk_blurred_images, dim=0).to(device)

        # Compute the absolute difference with the original image
        diff = torch.abs(chunk_blurred_images_tensor - original_image_tensor)

        # Move diff to CPU and compute the minimum difference and corresponding depth
        diff_cpu = diff.cpu().squeeze()  # Remove extra dimensions to match (height, width)
        for j in range(diff_cpu.shape[0]):
            mask = diff_cpu[j] < min_diff  # Find where the current slice has the minimum difference
            min_diff[mask] = diff_cpu[j][mask]  # Update the minimum difference
            depth_map[mask] = i + j  # Update the depth map with the slice index

    return depth_map


# Compute the depth map in chunks to avoid memory overflow
depth_map_tensor = estimate_depth_from_blur_in_chunks(blurred_images_cpu, input_image_padded, chunk_size=10)

# Remove padding from depth map to restore the original size
depth_map_tensor = depth_map_tensor[128:-128, 128:-128]

# Move depth map back to CPU for visualization
depth_map = depth_map_tensor.cpu().numpy()

# Step 6: Display the estimated depth map
plt.imshow(depth_map, cmap='plasma')
plt.title('Estimated Depth Map (GPU Accelerated with FFT and Padding)')
plt.colorbar()
plt.show()

# Optionally, save the depth map as a new TIFF image
output_path = 'depth_map_gpu_fft_padded.tif'
tiff.imwrite(output_path, depth_map.astype(np.uint8))

print(f"Depth map saved to: {output_path}")
