import os
from scipy.io import loadmat
import numpy as np
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def process_single_file(filename, input_folder, output_folder):
    mat_file_path = os.path.join(input_folder, filename)

    # Load .mat file
    data = loadmat(mat_file_path)
    # Assume the variable name to save is 'ObjRecon'
    image_data = data['ObjRecon']

    # Convert to float32 and adjust dimensions
    image = image_data.astype(np.float32)
    image = image.transpose(2, 0, 1)

    # Build output file path
    output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.tif')

    # Save as TIFF
    tiff.imwrite(output_file_path, image)
    return output_file_path


def process_mat_files(input_folder, output_folder, max_workers=4):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of .mat files
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]

    # Multi-threading with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Using a dictionary to keep track of file processing futures
        futures = {executor.submit(process_single_file, filename, input_folder, output_folder): filename for filename in
                   mat_files}

        # Progress bar
        for future in tqdm(as_completed(futures), total=len(futures)):
            filename = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Input and output folder paths
input_folder = 'anonymousVCD_dataset/fixed_fish/debug/Green/Red_Recon'
output_folder = 'anonymousVCD_dataset/fixed_fish/debug/Green/gt'

# Process .mat files in multiple threads
process_mat_files(input_folder, output_folder, max_workers=8)
