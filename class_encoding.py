import numpy as np
import os

# Base directory where the results folders are located
base_dir = path/to/kidney_preprocessed

# Iterate through each results directory
for i in range(0, 1982):  # Range is inclusive of 1981
    results_dir = os.path.join(base_dir, f"results{i}")
    # Assuming there's a corresponding subdirectory named 'resultX' for each 'resultsX'
    subdirectory = os.path.join(results_dir, f"results{i}")
    
    # Check if the subdirectory exists
    if os.path.exists(subdirectory) and os.path.isdir(subdirectory):
        filenames = [f for f in os.listdir(subdirectory) if f.startswith("segment") and f.endswith(".npy")]

        # Initialize an empty list to store one-hot encoded arrays
        one_hot_encodings = []

        for filename in filenames:
            # Extract the class number from the filename, which follows 'class_' and precedes '.npy'
            class_number_str = filename.split('_class_')[-1].split('.')[0]  # Extracts the part after 'class_' and before '.npy'
            class_number = int(class_number_str)
            
            # Validate the class number is within the expected range (1 to 4)
            print(f"Class number {class_number} for file {filename} will be saved into {results_dir}")
            if 1 <= class_number <= 4:
                # Create a one-hot encoded array with 4 zeros
                one_hot = np.zeros(4)
                # Set the index corresponding to class_number - 1 to 1
                one_hot[class_number - 1] = 1
                one_hot_encodings.append(one_hot)
            else:
                raise ValueError(f"Class number {class_number} out of expected range [1, 4] for file {filename}")

        # Convert the list of one-hot encodings to a numpy array
        class_encodings = np.array(one_hot_encodings)
        
        # Save the numpy array to the class_encoding.npy file directly in the 'resultsX' directory
        np.save(os.path.join(results_dir, "class_encoding.npy"), class_encodings)

print("Class encoding files created successfully.")
