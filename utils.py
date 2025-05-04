import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)



def visualize_datasets(images, labels, k=4, cols=4):
    """
    Visualizes a grid of images with their labels.

    Args:
        images (list or np.ndarray): List or array of images (expected to be in [0, 255] range).
        labels (list): List of corresponding labels.
        k (int): Number of images to display.
        cols (int): Number of columns in the grid.
    """
    if k > len(images):
        k = len(images)
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(6 * cols, 4 * rows))

    for i in range(k):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.title.set_text(labels[i])
        # Display scaled if needed, assuming input images are 0-255 for visualization
        plt.imshow(images[i] / 255.0)
        plt.colorbar()  # Added colorbar back as in original
        plt.axis("off")

    plt.tight_layout()  # Improve spacing
    plt.show(block=False)


def plot_acc(model_history, name, save_path=None):
    """
    Plots training and validation accuracy over epochs.

    Args:
        model_history (History): Keras History object.
        name (str): Name for the plot title.
        save_path (str, optional): Path to save the plot.
    """
    plt.rcParams.update({"font.size": 14})
    print(f"\nPlotting Accuracy for {name}")
    epochs = len(model_history.history["accuracy"])
    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(0, epochs),
        model_history.history["accuracy"],
        label="train_acc",
        marker="o",
    )
    plt.plot(
        np.arange(0, epochs),
        model_history.history["val_accuracy"],
        label="val_acc",
        marker="o",
    )
    plt.title(f"Training Accuracy - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)


def plot_loss(model_history, name, save_path=None):
    """
    Plots training and validation loss over epochs.

    Args:
        model_history (History): Keras History object.
        name (str): Name for the plot title.
        save_path (str, optional): Path to save the plot.
    """
    plt.rcParams.update({"font.size": 14})
    print(f"\nPlotting Loss for {name}")
    epochs = len(model_history.history["loss"])
    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(0, epochs),
        model_history.history["loss"],
        label="train_loss",
        marker="o",
    )
    plt.plot(
        np.arange(0, epochs),
        model_history.history["val_loss"],
        label="val_loss",
        marker="o",
    )
    plt.title(f"Training Loss - {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)


def plot_confusion_matrix(
    cm,
    classes,
    normalize=True,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        cm (np.ndarray): Confusion matrix array.
        classes (list): List of class names.
        normalize (bool): Whether to normalize the matrix.
        title (str): Title of the plot.
        cmap (matplotlib.colors.Colormap): Colormap for the plot.
        save_path (str, optional): Path to save the plot.
    """
    plt.rcParams.update({"font.size": 12})  # Smaller font for potentially many classes
    print(f"\nPlotting Confusion Matrix: {title}")

    class_count = len(classes)
    if normalize:
        cm = cm.astype("float") / (
            cm.sum(axis=1)[:, np.newaxis] + 1e-6
        )  # Add small epsilon to avoid division by zero
        cm = np.round(cm, 2)
        fmt = "g"  # Use general format for normalized values
    else:
        fmt = "d"  # Use integer format for raw counts

    plt.figure(figsize=(10, 8))  # Adjusted size slightly
    sns.heatmap(
        cm, annot=True, vmin=0, fmt=fmt, cmap=cmap, cbar=True if normalize else False
    )  # Show cbar for normalized
    plt.xticks(np.arange(class_count) + 0.5, classes, rotation=90)
    plt.yticks(np.arange(class_count) + 0.5, classes, rotation=0)
    plt.xlabel("Predicted Label")  # More descriptive label
    plt.ylabel("True Label")  # More descriptive label
    plt.title(title)
    plt.tight_layout()  # Improve spacing
    if save_path:
        plt.savefig(save_path)
    plt.show(block=False)


def save_history(history, path):
    """
    Saves training history from a Keras History object to a CSV file.

    Args:
        history (History): Keras History object.
        path (str): Path to save the CSV file.
    """
    history_df = pd.DataFrame(history.history)
    history_df.index.name = "Epoch"
    history_df.to_csv(path)
    print(f"Saved history to {path}")


# Added KMeans function if needed, but won't be run in main
def plot_kmeans_segmentation(image, n_clusters=2):
    """
    Performs KMeans clustering on an image and visualizes the segmentation.

    Args:
        image (np.ndarray): Input image (expected to be in [0, 255] range, RGB).
        n_clusters (int): Number of clusters for KMeans.
    """
    print(f"\nPerforming KMeans segmentation with {n_clusters} clusters...")
    w, h, channel = image.shape
    # Reshape image to be a list of pixels
    X = np.reshape(image, (w * h, channel)).astype(
        np.float32
    )  # Convert to float for KMeans

    # Kmeans clustering
    try:
        # Use cv2.kmeans which is often faster for images
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, pred_label, centers = cv2.kmeans(
            X, n_clusters, None, criteria, 10, flags
        )
        pred_label = pred_label.reshape((h, w))

    except Exception as e:
        print(f"Error during KMeans: {e}")
        print("Falling back to sklearn.cluster.KMeans (might be slower)...")
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X)
            pred_label = kmeans.labels_.reshape((h, w))
            centers = kmeans.cluster_centers_
        except Exception as e2:
            print(f"Error with sklearn KMeans: {e2}. Skipping segmentation plot.")
            return  # Skip plot if clustering fails

    # Create mask visualization
    mask = np.zeros(shape=(h, w, channel), dtype=np.float32)
    # Use cluster centers as colors, scaled to [0, 1]
    scaled_centers = centers / 255.0 if centers.max() > 1.0 else centers
    # Ensure centers are in RGB order if input was RGB
    mask_colors = [tuple(c) for c in scaled_centers]

    for i in range(n_clusters):
        mask[pred_label == i] = mask_colors[
            i % len(mask_colors)
        ]  # Use modulo in case n_clusters > len(mask_colors)

    # Optional: Apply morphology (as in original notebook)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    # Display image clustering
    fg, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].imshow(image / 255.0)  # Display original scaled [0, 1]
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask)
    ax[1].set_title(f"Segmented Mask ({n_clusters} clusters)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show(block=False)


def visualize_predictions(images, true_labels, predicted_probs, classes, k=5):
    """
    Visualizes sample test images, their true labels, and predicted probabilities.

    Args:
        images (np.ndarray): Array of test images (expected scaled [0, 1] or [0, 255]).
        true_labels (np.ndarray): One-hot encoded true labels.
        predicted_probs (np.ndarray): Array of predicted probabilities per class.
        classes (list): List of class names.
        k (int): Number of samples to visualize.
    """
    plt.rcParams.update({"font.size": 14})  # Adjusted font size
    print(f"\nVisualizing {k} sample test predictions:")

    n_samples = images.shape[0]
    sample_indices = np.random.choice(
        range(n_samples), min(k, n_samples), replace=False
    )  # Ensure no replacement if k > n_samples

    x_sample = images[sample_indices]
    y_sample_true_onehot = true_labels[sample_indices]
    y_sample_pred_probs = predicted_probs[sample_indices]

    y_sample_true_indices = np.argmax(y_sample_true_onehot, axis=1)
    y_sample_true_names = [classes[i] for i in y_sample_true_indices]

    # Helper to format labels for the plot
    def format_label(label):
        return "\\n".join(label.split())

    short_labels = list(map(format_label, classes))
    bar_colors = plt.cm.viridis(
        np.linspace(0, 1, len(classes))
    )  # Use a colormap for bars

    fig, ax = plt.subplots(
        len(sample_indices), 2, figsize=(18, 4 * len(sample_indices))
    )  # Adjust figure size

    for i in range(len(sample_indices)):
        # Display image (assuming it needs to be scaled to [0, 1] for imshow)
        # If images are already scaled to [0, 1], remove the / 255.0
        ax[i, 0].imshow(
            x_sample[i] if np.max(x_sample[i]) <= 1.0 else x_sample[i] / 255.0
        )
        ax[i, 0].axis("off")

        # Plot predicted probabilities
        ax[i, 1].bar(short_labels, y_sample_pred_probs[i] * 100, color=bar_colors)
        ax[i, 1].set_ylabel("Probability (%)")
        predicted_class_index = np.argmax(y_sample_pred_probs[i])
        predicted_class_name = classes[predicted_class_index]
        ax[i, 1].set_title(
            f"True: {y_sample_true_names[i]} | Predicted: {predicted_class_name}"
        )
        ax[i, 1].set_ylim(0, 100)  # Set y-limit to 0-100% for clarity

    plt.tight_layout()
    plt.show(block=False)


if __name__ == "__main__":
    # Example KMeans Usage (using a dummy image)
    dummy_img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    plot_kmeans_segmentation(dummy_img, n_clusters=3)
