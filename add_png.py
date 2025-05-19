import os

# Specify the directory containing the files to be renamed
directory = 

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Construct the original full file path
    original_file_path = os.path.join(directory, filename)
    
    # Check if it's a file and not a directory
    if os.path.isfile(original_file_path):
        # Construct new file name by appending .png to the original file name
        new_file_name = f"{filename}.png"
        # Construct the new full file path
        new_file_path = os.path.join(directory, new_file_name)
        
        # Rename the file
        os.rename(original_file_path, new_file_path)

print("All files have been renamed successfully.")
