import os
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
from tqdm import tqdm

from config import cfg

def load_images(metadata_path, image_dir_path, classes, img_size, max_original_per_class):
    """
    Loads and preprocesses images based on metadata, limiting images per class.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        image_dir_path (str): Path to the directory containing images.
        classes (list): List of class names to include.
        img_size (int): Target size for resizing images.
        max_original_per_class (int): Maximum number of original images to load per class.

    Returns:
        tuple: (list of loaded images, list of corresponding labels,
                dict of image lists per class, dict of counts per class)
    """
    metadata_df = pd.read_csv(metadata_path)

    error_count = 0
    data_dict = defaultdict(list)
    count_dict = defaultdict(int)
    root_images = []
    root_labels = []

    print(f"Loading and preprocessing images (limit {max_original_per_class} per class)...")
    # Ensure correct file extension for images
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))}


    for i in tqdm(metadata_df.index, desc="Loading images"):
        row = metadata_df.loc[i]
        label = row.dx
        image_id = row.image_id

        if label not in classes:
            continue

        if count_dict[label] >= max_original_per_class:
            continue

        # Construct image path using the file name derived from image_id
        image_filename = image_files.get(image_id)
        if image_filename is None:
            #print(f"Warning: Image file not found for ID {image_id}. Skipping.") # Too verbose
            error_count += 1
            continue

        path = os.path.join(image_dir_path, image_filename)

        try:
            image = cv2.imread(path)
            if image is None:
                 #print(f"Warning: Could not read image file {path}. Skipping.") # Too verbose
                 error_count += 1
                 continue

            # Resize and color conversion
            image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            count_dict[label] += 1
            data_dict[label].append(image)
            root_images.append(image)
            root_labels.append(label)
        except Exception as e:
            # print(f"Error processing image {path}: {e}. Skipping.") # Too verbose
            error_count += 1

    print(f"Finished loading images.")
    print(f"ErrorCount (files not found or read errors): {error_count}")
    print(f"Total original images loaded: {len(root_images)}")

    return root_images, root_labels, data_dict, count_dict

if __name__ == '__main__':
    # Example usage:
    images, labels, data_dict, count_dict = load_images(
        cfg.METADATA_PATH,
        cfg.IMAGE_DIR_PATH,
        cfg.CLASSES,
        cfg.IMG_SIZE,
        cfg.MAX_ORIGINAL_PER_CLASS
    )

    print("\nCounts of original images loaded per class:")
    label_counts = pd.Series(labels).value_counts()
    print(label_counts)

    # Optional: Visualize original data samples
    from utils import visualize_datasets
    print("\nVisualizing sample original images:")
    sample_images = [data_dict[lab][0] for lab in cfg.CLASSES if lab in data_dict]
    sample_labels = [lab for lab in cfg.CLASSES if lab in data_dict]
    visualize_datasets(sample_images, sample_labels, k=len(sample_images), cols=3)