import os

# Specify the directory and prefix
directory = "anonymousVCD_dataset/NemoS/test/g"
prefix = "Nemos5v_"

# Check if the directory exists
if os.path.exists(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .tif extension
        if filename.endswith(".tif"):
            # Define the new filename with the prefix added
            new_filename = prefix + filename
            # Get full file paths
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_filepath, new_filepath)
    print(f"All .tif files in {directory} have been renamed with the prefix '{prefix}'.")
else:
    print(f"The directory {directory} does not exist.")
