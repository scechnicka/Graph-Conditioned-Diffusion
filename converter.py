import numpy as np
import matplotlib.pyplot as plt

# Define the image path
img_path = path/to/resultsX_feature_vectors.npy
image_data = np.load(img_path)
# Load the image data from the .npz file
print(image_data)

# Display the image using plt.imshow
plt.imshow(image_data)  # You can adjust the cmap if needed
plt.axis('off')  # Turn off axes
plt.show()
plt.savefig('temp.png', format='png')

