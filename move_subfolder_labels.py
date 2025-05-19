import os
import shutil

source_dir = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/copy_paste/image_labels_npy"
destination_dir = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/copy_paste/image_labels_total"

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through each folder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    # Check if it is indeed a folder
    if os.path.isdir(folder_path):
        # Extract the number (X) from the folder name ('resultsX')
        match = folder_name.split('results')
        if len(match) > 1:
            number = match[1]
            
            # For each file in the folder, check if it matches the required pattern
            for file_name in os.listdir(folder_path):
                if file_name.endswith('_OUT.npy'):  # More flexible check for files ending with '_OUT.npy'
                    # Extract the digit (0 or 1) from the file name
                    digit = file_name.split('_')[1]
                    # Construct new file name including the "_OUT" part
                    new_file_name = f"sample_{number}_{digit}_OUT.npy"
                    # Construct full source and destination paths
                    source_file_path = os.path.join(folder_path, file_name)
                    destination_file_path = os.path.join(destination_dir, new_file_name)
                    
                    # Move and rename the file
                    shutil.move(source_file_path, destination_file_path)

print("Files have been successfully moved and renamed.")
