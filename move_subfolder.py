import os
import shutil

# Define the source and destination directories
source_dir = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/copy_paste_short/manual"
destination_dir = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/copy_paste_short/manual_total"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate over all directories and subdirectories in the source directory
for root, dirs, files in os.walk(source_dir):
    for dir_name in dirs:
        #if dir_name.startswith('partial_') and 'graph.npy' in dir_name:
        if dir_name.startswith('conditioning_') and '.npy' in dir_name:
            # Extract the X value from the directory name
            parts = dir_name.split('_')
            if len(parts) < 4:
                continue  # Skip this directory if it doesn't match the expected pattern
            X = parts[1]
            # Construct the full path to this directory
            full_dir_path = os.path.join(root, dir_name)
            # Iterate over all files in the directory
            for file_name in os.listdir(full_dir_path):
                if file_name.endswith('.png') and file_name.startswith('sample_'):
                    # Extract the A value from the file name
                    A = file_name.split('_')[1].split('.')[0]
                    # Construct the new file name
                    new_file_name = f"sample_{X}_{A}.png"
                    # Move and rename the file
                    shutil.move(os.path.join(full_dir_path, file_name), os.path.join(destination_dir, new_file_name))

print("Files have been moved and renamed successfully.")
