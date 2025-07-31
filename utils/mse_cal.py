import numpy as np
import tifffile

def calculate_mse(tif_path1, tif_path2):
    """
    Calculate the Mean Squared Error between two normalized TIFF files.

    Parameters:
    tif_path1 (str): Path to the first TIFF file.
    tif_path2 (str): Path to the second TIFF file.

    Returns:
    float: The Mean Squared Error between the two normalized images.
    """
    # Read the TIFF files
    image1 = tifffile.imread(tif_path1).astype(np.float32)
    image2 = tifffile.imread(tif_path2).astype(np.float32)

    # Ensure the images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("The input TIFF files must have the same dimensions.")

    # Normalize the images to the range [0, 1]
    image1_normalized = (image1 - np.min(image1)) / (np.max(image1) - np.min(image1))
    image2_normalized = (image2 - np.min(image2)) / (np.max(image2) - np.min(image2))

    # Calculate the Mean Squared Error
    mse = np.mean((image1_normalized - image2_normalized) ** 2)

    return mse

# Example usage:
mse = calculate_mse('/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt_RLD100_900/00000023.tif',
                    '/gpfs/home/LifeSci/wenlab/hefengcs/VCD_dataset/gt300/00000023.tif')
print(f"The Mean Squared Error is: {mse}")
