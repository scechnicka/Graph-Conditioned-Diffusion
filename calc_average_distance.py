import os

# Assuming the folder path is given or known, let's use a mock path for demonstration purposes
folder_path = path/to/diffusion_model_graph_outputs/interpolated/manual

# This function extracts the X and Y values from the subfolder names and calculates their averages
def calculate_average_xy(folder_path):
    x_values = []
    y_values = []

    # List all items in the given folder path
    for item in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, item)):  # Check if the item is a directory
            # Try to extract X and Y from the folder name according to the given pattern
            try:
                parts = item.split('_')
                x = parts[2]
                y = parts[3]
                x = float(x)
                y = float(y)
                x_values.append(x)
                y_values.append(y)
            except (IndexError, ValueError):
                # Skip the item if it doesn't match the expected pattern or if conversion to int fails
                continue

    # Calculate the average of X and Y values if there are any valid folders found
    if x_values and y_values:
        average_x = sum(x_values) / len(x_values)
        average_y = sum(y_values) / len(y_values)
    else:
        average_x = average_y = None

    return average_x, average_y
# Now, calculate the averages
combined_score, average_spectral_distance = calculate_average_xy(folder_path)
print(f"Combined score (spectral + feature): {combined_score}, Spectral score : {average_spectral_distance} ")
