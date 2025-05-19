import os
import shutil
import re 

# Source directory where the subfolders are located
source_dir = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/modified/image/changed_labels"

# Destination directory where you want to move and rename the files
dest_dir = "/home/atuin/b143dc/b143dc22/GCD/Diffusion_model_graph_outputs/modified/image/changed_labels/total/"

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Loop through each subdirectory in the source directory
for subdir in os.listdir(source_dir):
    # Check if the subdirectory matches the pattern "partial_X_Y_Z_graph"
    if subdir.startswith("conditioning_") and subdir.endswith("_1024"):
        # Extract X from the subdirectory name
        #x = subdir.split('_')[1] # for normal files not small mods
        match = re.search(r'results(\d+)', subdir) #for small mods
        if match: #small mods
            x = match.group(1)  # small mods
        sub_path = os.path.join(source_dir, subdir)
        # Loop through each file in the subdirectory
        for file in os.listdir(sub_path):
            if file.startswith("sample_") and file.endswith(".npy"):
                # Extract A from the file name
                a = file.split('_')[1].split('.')[0]
                # Define the new file name with the format "sample_X_A_OUT.npy"
                new_file_name = f"sample_{x}_{a}_OUT.npy"
                # Define the source and destination file paths
                src_file_path = os.path.join(sub_path, file)
                dest_file_path = os.path.join(dest_dir, new_file_name)
                # Move and rename the file
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved and renamed {file} to {new_file_name}")

print("All files have been moved and renamed successfully.")
