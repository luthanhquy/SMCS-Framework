import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras_cv

from config import cfg
from utils import visualize_datasets


def balance_and_augment_with_mixup(
    root_images,
    root_labels,
    data_dict,
    count_dict,
    target_total_per_class,
    mix_per_batch,
    img_size,
    num_classes,
    classes,
):
    """
    Balances the dataset using MixUp augmentation.

    Args:
        root_images (list): List of original loaded images.
        root_labels (list): List of original corresponding labels.
        data_dict (dict): Dictionary of original image lists per class.
        count_dict (dict): Dictionary of original counts per class.
        target_total_per_class (int): Target number of images per class after MixUp.
        mix_per_batch (int): Batch size for MixUp generation.
        img_size (int): Image size.
        num_classes (int): Number of classes.
        classes (list): List of class names.

    Returns:
        tuple: (numpy array of merged images, numpy array of merged one-hot labels)
    """
    print("\nApplying MixUp augmentation to balance dataset...")

    # Convert original data to arrays and tensors
    x_arr_orig = np.array(root_images)
    # Convert labels to one-hot encoding based on the defined classes order
    y_arr_orig = tf.one_hot(
        [classes.index(label) for label in root_labels], num_classes
    ).numpy()

    x_tensor_orig = tf.constant(x_arr_orig, dtype=tf.float32)
    y_tensor_orig = tf.constant(y_arr_orig, dtype=tf.float32)

    mixup_layer = keras_cv.layers.Augmenter([keras_cv.layers.MixUp()])

    x_tensor_mix = None
    y_tensor_mix = None

    # Use a DataFrame to quickly get indices for sampling
    root_label_df = pd.DataFrame({"dx": root_labels})

    for label in classes:
        count = count_dict.get(label, 0)
        num_mix_needed = target_total_per_class - count

        if num_mix_needed <= 0:
            print(
                f"Class '{label}': Already {count} images, no MixUp needed (target {target_total_per_class})."
            )
            continue

        # Calculate the number of batches needed for MixUp
        num_batches = num_mix_needed // mix_per_batch
        num_mix_generated = num_batches * mix_per_batch

        if num_batches == 0:
            print(
                f"Class '{label}': Found {count} original images. Needs {num_mix_needed} more but less than mix_per_batch ({mix_per_batch}). No MixUp generated."
            )
            continue

        print(
            f"Class '{label}': Found {count} original image(s). Generating {num_mix_generated} ({num_batches} batch(s)) via MixUp to reach target {target_total_per_class}..."
        )

        # Find indices of original images for this class
        original_indices_for_class = root_label_df[
            root_label_df.dx == label
        ].index.tolist()

        for i in range(num_batches):
            # Sample indices *with replacement* from the original images of this class
            # Mixing images from the same class with each other is a common strategy
            # If you want to mix images from different classes, the MixUp layer handles it
            # when given a batch with varied labels.
            # Here, we sample twice to get two images for mixing.
            # The keras_cv.layers.MixUp layer itself takes a batch and pairs them up internally.
            # So we just need to provide a batch of images from the class we want to augment.
            if len(original_indices_for_class) < mix_per_batch:
                # If not enough original images, sample with replacement
                sampled_indices = np.random.choice(
                    original_indices_for_class, mix_per_batch, replace=True
                )
            else:
                # Sample without replacement if enough original images
                sampled_indices = np.random.choice(
                    original_indices_for_class, mix_per_batch, replace=False
                )

            x_mix_batch = tf.gather(x_tensor_orig, sampled_indices)
            y_mix_batch = tf.gather(y_tensor_orig, sampled_indices)

            # Apply MixUp to the batch
            # Keras-CV's MixUp layer requires input as a dictionary {'images': ..., 'labels': ...}
            mix_out = mixup_layer({"images": x_mix_batch, "labels": y_mix_batch})

            if x_tensor_mix is None:
                x_tensor_mix = mix_out["images"]
                y_tensor_mix = mix_out["labels"]
            else:
                x_tensor_mix = tf.concat([x_tensor_mix, mix_out["images"]], axis=0)
                y_tensor_mix = tf.concat([y_tensor_mix, mix_out["labels"]], axis=0)

        print(
            f"Mixup complete for '{label}'. Original: {count}, Generated: {num_mix_generated}, Total: {count + num_mix_generated}."
        )

    if x_tensor_mix is not None:
        x_merge_mix = tf.concat([x_tensor_orig, x_tensor_mix], axis=0).numpy()
        y_merge_mix = tf.concat([y_tensor_orig, y_tensor_mix], axis=0).numpy()
    else:
        # No MixUp was needed
        x_merge_mix = x_arr_orig
        y_merge_mix = y_arr_orig

    print(f"\nTotal images after merging original and mixed: {x_merge_mix.shape[0]}")

    # Print counts after balancing
    mixed_labels_indices = np.argmax(y_merge_mix, axis=1)
    mixed_labels_names = [classes[i] for i in mixed_labels_indices]
    print("\nCounts per class after MixUp:")
    print(pd.Series(mixed_labels_names).value_counts())

    return x_merge_mix, y_merge_mix


def split_data(x, y, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits data into training, validation, and test sets with stratification.

    Args:
        x (np.ndarray): Image data array.
        y (np.ndarray): One-hot encoded label array.
        test_size (float): Proportion of data for the test set.
        val_size (float): Proportion of data for the validation set (from the remaining data).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (x_train, x_val, x_test, y_train, y_val, y_test)
    """
    print(
        f"\nSplitting data: Train ({1 - test_size - val_size:.2f}) / Val ({val_size:.2f}) / Test ({test_size:.2f})"
    )
    # Split the data into training and remaining sets (validation + test)
    # Stratify based on the original class index (derived from one-hot)
    y_indices = np.argmax(y, axis=1)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=y_indices,
    )

    # Split the remaining data into validation and test sets
    # Stratify based on the original class index (derived from one-hot)
    y_temp_indices = np.argmax(y_temp, axis=1)
    test_proportion_of_temp = test_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_proportion_of_temp,
        random_state=random_state,
        stratify=y_temp_indices,
    )

    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return x_train, x_val, x_test, y_train, y_val, y_test


def scale_images(images, scale_range=(0.0, 1.0)):
    """
    Scales image pixel values to a specified range.

    Args:
        images (np.ndarray): Image data array (e.g., uint8).
        scale_range (tuple): The target range (min, max).

    Returns:
        np.ndarray: Scaled image data array (float32).
    """
    min_val, max_val = scale_range
    # Convert to float32
    images = images.astype(np.float32)
    # Simple min-max scaling
    images = images * (max_val - min_val) / 255.0 + min_val
    return images


if __name__ == "__main__":
    # Example Usage:
    from data_loader import load_images

    root_images, root_labels, data_dict, count_dict = load_images(
        cfg.METADATA_PATH,
        cfg.IMAGE_DIR_PATH,
        cfg.CLASSES,
        cfg.IMG_SIZE,
        cfg.MAX_ORIGINAL_PER_CLASS,
    )

    x_merged, y_merged = balance_and_augment_with_mixup(
        root_images,
        root_labels,
        data_dict,
        count_dict,
        cfg.TARGET_TOTAL_PER_CLASS,
        cfg.MIX_PER_BATCH,
        cfg.IMG_SIZE,
        cfg.NUM_CLASSES,
        cfg.CLASSES,
    )

    # Scale the data after merging and before splitting (or after splitting train/val/test)
    # Scaling before splitting ensures consistency.
    x_scaled = scale_images(x_merged, scale_range=(0.0, 1.0))  # Scale to [0, 1]

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x_scaled, y_merged)

    # Optional: Visualize MixUp samples from the merged data
    print("\nVisualizing sample MixUp images from the merged dataset:")
    mix_length = x_merged.shape[0]
    sample_indices = np.random.choice(
        mix_length, size=min(9, mix_length)
    )  # Don't request more than available
    sample_images_display = tf.gather(
        x_merged, sample_indices
    ).numpy()  # Use original values for display
    sample_labels_onehot = tf.gather(y_merged, sample_indices).numpy()

    # For display purposes, pick the class with the highest weight
    sample_labels_indices = np.argmax(sample_labels_onehot, axis=1)
    sample_labels_names = [cfg.CLASSES[i] for i in sample_labels_indices]

    visualize_datasets(
        sample_images_display, sample_labels_names, k=len(sample_indices), cols=3
    )

    # Check scaled data type and range (should be float32 and within [0, 1])
    print(f"\nScaled data dtype: {x_train.dtype}")
    print(f"Scaled data min/max: {x_train.min()}/{x_train.max()}")
