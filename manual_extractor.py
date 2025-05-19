import numpy as np
import skimage.measure
from scipy import ndimage
import glob
import os


def load_data(npz_path):
    data = np.load(npz_path)
    image = data['image']
    label_mask = data['label']
    return image, label_mask


def normalize_and_prepare_features_1024(features, desired_length):
    normalized_features = {
        'Centroid_x': features['Centroid'][0] / 1024,  # Normalize x coordinate of centroid to image width (1024)
        'Centroid_y': features['Centroid'][1] / 1024,  # Normalize y coordinate of centroid to image height (1024)
        'Average Area': features['Average Area'] / (1024 * 1024),  # Normalize area to the maximum possible value (1024*1024)
        'Bounding Box_x': features['Bounding Box'][0] / 1024,  # Normalize x coordinate of bounding box to image width (1024)
        'Bounding Box_y': features['Bounding Box'][1] / 1024,  # Normalize y coordinate of bounding box to image height (1024)
        'Bounding Box_width': features['Bounding Box'][2] / 1024,  # Normalize width of bounding box to image width (1024)
        'Bounding Box_height': features['Bounding Box'][3] / 1024  # Normalize height of bounding box to image height (1024)
    }

    # Concatenate normalized features into a single vector
    feature_vector = []
    for key in sorted(normalized_features.keys()):
        feature_vector.append(normalized_features[key])
    # One-hot encode the class
    class_one_hot = np.zeros(4)
    class_one_hot[features['Class'] - 1] = 1  # Assuming class ranges from 1 to 4

    # Concatenate normalized features and class one-hot encoding into a single vector
    feature_vector = np.concatenate((class_one_hot, list(normalized_features.values())))

    # Pad or truncate feature vector to the desired length
    if len(feature_vector) < desired_length:
        feature_vector = np.pad(feature_vector, (0, desired_length - len(feature_vector)), mode='constant')
    return np.array(feature_vector)

def normalize_and_prepare_features(features):
    normalized_features = {
        'Centroid_x': features['Centroid'][0] / 1024,  # Normalize x coordinate of centroid to image width (1024)
        'Centroid_y': features['Centroid'][1] / 1024,  # Normalize y coordinate of centroid to image height (1024)
        'Average Area': features['Average Area'] / (1024 * 1024),  # Normalize area to the maximum possible value (1024*1024)
        'Bounding Box_x': features['Bounding Box'][0] / 1024,  # Normalize x coordinate of bounding box to image width (1024)
        'Bounding Box_y': features['Bounding Box'][1] / 1024,  # Normalize y coordinate of bounding box to image height (1024)
        'Bounding Box_width': features['Bounding Box'][2] / 1024,  # Normalize width of bounding box to image width (1024)
        'Bounding Box_height': features['Bounding Box'][3] / 1024  # Normalize height of bounding box to image height (1024)
    }

    # Concatenate normalized features into a single vector
    feature_vector = []
    for key in sorted(normalized_features.keys()):
        feature_vector.append(normalized_features[key])
     # One-hot encode the class
    class_one_hot = np.zeros(4)
    class_one_hot[features['Class'] - 1] = 1  # Assuming class ranges from 1 to 4

    # Concatenate normalized features and class one-hot encoding into a single vector
    feature_vector = np.concatenate((class_one_hot, list(normalized_features.values())))
    return np.array(feature_vector)
    

def calculate_features_and_vectors(label_mask, desired_vector_length):
    instance_labels = np.zeros_like(label_mask)
    features_list = []
    vectors_list = []
    vectors_list_1024 = []
    first_label = 0
    total_instances = 0  # Variable to count total instances processed


    for k in range(1, 5):
        l = (label_mask == k).astype(np.uint8)
        label, nb = ndimage.label(l)
        for i in range(1, np.max(label) + 1):
            blob_mask = (label == i)
            if np.sum(blob_mask) < 5:  # Skip blobs with fewer than 5 pixels
                continue

            region_props = skimage.measure.regionprops(blob_mask.astype(int))[0]
            inst_id = i + first_label
            instance_labels[blob_mask] = inst_id

            # Extract raw features
            raw_features = {
                'ID': inst_id,
                'Class': k,
                'Centroid': region_props.centroid,
                'Average Area': region_props.area,
                'Bounding Box': region_props.bbox
            }
            
            # Calculate and normalize the feature vector 1024
            feature_vector_1024 = normalize_and_prepare_features_1024(raw_features, desired_vector_length)
            
            # Calculate and normalize the feature vector
            feature_vector = normalize_and_prepare_features(raw_features)
            # Append to lists
            features_list.append(raw_features)
            vectors_list_1024.append(feature_vector_1024)
            vectors_list.append(feature_vector)
            total_instances += 1  # Increment total instances counter
  
                
        first_label += nb

    return features_list, vectors_list_1024,vectors_list

desired_vector_length = 1024  # Adjust this to match your model's input size
def save_features(features, filename):
    with open(filename, 'w') as f:
        for feature in features:
            f.write(f"ID: {feature['ID']}, Class: {feature['Class']}, Centroid: {feature['Centroid']}, "
                    f"Average Area: {feature['Average Area']}, Bounding Box: {feature['Bounding Box']}\n")


input_folder = path/to/folder/with_data
base_output_folder = where/you/want/to/save_data
npz_files = glob.glob(os.path.join(input_folder, '*.npz'))
total_files = len(npz_files)
processed_files = 0

for npz_file in npz_files:
    image, label_mask = load_data(npz_file)
    features, feature_vectors_1024,feature_vectors = calculate_features_and_vectors(label_mask, desired_vector_length)

    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    specific_output_folder = os.path.join(base_output_folder, base_name)
    os.makedirs(specific_output_folder, exist_ok=True)

    # Save features text
    output_filename = os.path.join(specific_output_folder, f"{base_name}_features_manual.txt")
    save_features(features, output_filename)

    # Save feature vectors
    vectors_filename_1024 = os.path.join(specific_output_folder, f"{base_name}_feature_vectors_manual_1024.npy")
    np.save(vectors_filename_1024, np.array(feature_vectors_1024))
    
    # Save feature vectors with shape (number_of_nodes, number_of_features)
    vectors_filename = os.path.join(specific_output_folder, f"{base_name}_feature_vectors_manual.npy")
    np.save(vectors_filename, np.array(feature_vectors))
    processed_files += 1
    print(f"Processed {processed_files} out of {total_files} files.")
