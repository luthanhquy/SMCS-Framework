import os
import numpy as np
import pandas as pd
import tensorflow as tf

from config import cfg
from data_loader import load_images
from data_processor import balance_and_augment_with_mixup, split_data, scale_images
from models import build_model
from train import run_training_pipeline
from evaluate import (
    print_classification_report,
    calculate_metrics,
)  # plot_confusion_matrix imported in train.py
from explain import run_shap_explanation
from utils import (
    visualize_predictions,
    plot_kmeans_segmentation,
)  # plot_acc, plot_loss, save_history imported in train.py

# Set up GPU if available
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found. Using CPU.")

def predict_prob(model, x_test, batch_size):
    return model.predict(x_test, batch_size, verbose=1)

def main():
    print("--- Skin Cancer Classification Project ---")

    # --- 1. Load and Process Data ---
    # Load original images (potentially limited per class)
    root_images_orig, root_labels_orig, data_dict_orig, count_dict_orig = load_images(
        cfg.METADATA_PATH,
        cfg.IMAGE_DIR_PATH,
        cfg.CLASSES,
        cfg.IMG_SIZE,
        cfg.MAX_ORIGINAL_PER_CLASS,
    )

    # Perform MixUp augmentation to balance classes
    # This returns images in the original [0, 255] range
    x_merged, y_merged = balance_and_augment_with_mixup(
        root_images_orig,
        root_labels_orig,
        data_dict_orig,
        count_dict_orig,
        cfg.TARGET_TOTAL_PER_CLASS,
        cfg.MIX_PER_BATCH,
        cfg.IMG_SIZE,
        cfg.NUM_CLASSES,
        cfg.CLASSES,
    )

    # Scale images for model input (e.g., to [0, 1])
    # Apply scaling *after* merging but *before* splitting
    x_merged_scaled = scale_images(x_merged, scale_range=(0.0, 1.0))  # Scale to [0, 1]

    # Split the data into train, validation, and test sets
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        x_merged_scaled, y_merged, test_size=0.1, val_size=0.1
    )

    # --- Optional: Run KMeans Segmentation on a sample ---
    # Select a sample image from original loaded data (e.g., first image of first class)
    if data_dict_orig and list(data_dict_orig.keys()):
        first_class = list(data_dict_orig.keys())[0]
        if data_dict_orig[first_class]:
            sample_img_orig = data_dict_orig[first_class][0]
            print(
                f"\n--- Running KMeans Segmentation example on a sample from class '{first_class}' ---"
            )
            plot_kmeans_segmentation(sample_img_orig, n_clusters=2)
        else:
            print(
                "\nSkipping KMeans Segmentation example: No images loaded for the first class."
            )
    else:
        print("\nSkipping KMeans Segmentation example: No original images loaded.")

    # --- 2. Build and Train Models ---
    trained_models = {}
    for model_name in cfg.MODELS_TO_TRAIN:
        print(f"\n\n========== Start Process with model {model_name} ==========")

        # Build the model architecture
        model = build_model(model_name, cfg.INPUT_SHAPE, cfg.NUM_CLASSES)

        # Run the training pipeline (transfer learning + fine-tuning)
        final_model, y_pred_prob_test, y_pred_index_test, y_true_index_test = (
            run_training_pipeline(
                model, x_train, y_train, x_val, y_val, x_test, y_test, cfg
            )
        )
        trained_models[model_name] = final_model  # Store the final trained model

        # Print final classification report and metrics
        print(f"\n--- Final Results for {model_name} on Test Set ---")
        print_classification_report(y_true_index_test, y_pred_index_test, cfg.CLASSES)
        calculate_metrics(y_true_index_test, y_pred_index_test)  # Prints metrics

        print(f"\n========== Completed Process with model {model_name} ==========")

    # --- 3. Model Explanation (SHAP) ---
    # SHAP runs on the *original* loaded data (before MixUp) for visualization as in the notebook
    # If you want to explain predictions on the test set (which includes mixed images),
    # you would sample from x_test and y_test instead of data_dict_orig.
    # Keep original notebook logic for now.
    if trained_models:
        print("\n\n--- Running SHAP Explanations for Trained Models ---")
        for model_name, model in trained_models.items():
            print(f"\n--- Explaining {model_name} ---")
            run_shap_explanation(
                model, data_dict_orig, cfg.CLASSES, cfg, cfg.SHAP_NUM_SAMPLES_PER_CLASS
            )
    else:
        print("\n\nNo models were trained, skipping SHAP explanation.")

    # --- 4. Visualize Sample Test Predictions ---
    if trained_models and x_test is not None and y_test is not None:
        print("\n\n--- Visualizing Sample Test Predictions ---")
        for model_name, model in trained_models.items():
            print(f"\n--- Predictions for {model_name} ---")
            # Get predictions from the final trained model
            y_pred_prob_test = predict_prob(model, x_test, cfg.BATCH_SIZE)
            visualize_predictions(x_test, y_test, y_pred_prob_test, cfg.CLASSES, k=5)
    else:
        print(
            "\n\nNo models were trained or test data is missing, skipping prediction visualization."
        )

    print("\n--- Skin Cancer Classification Project Complete ---")


if __name__ == "__main__":
    main()
