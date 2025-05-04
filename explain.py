import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os

from config import cfg

# Assuming images are scaled to [0, 1] before being fed to the model for training
# The SHAP explainer should receive input data in the same format as the model expects.
# If using model-specific preprocessing (like preprocess_input), that should be applied here.
# If using [0,1] scaling, ensure the explainer gets [0,1] data.


def run_shap_explanation(model, data_dict, classes, config, num_samples_per_class=2):
    """
    Generates and plots SHAP explanations for sample images per class.

    Args:
        model (tf.keras.Model): The trained Keras model.
        data_dict (dict): Dictionary of original image lists per class (used for sampling).
        classes (list): List of class names.
        config (Config): Configuration object.
        num_samples_per_class (int): Number of samples to explain for each class.
    """
    print("\n--- Running SHAP Explanation ---")

    # Create the masker. Use "inpaint_telea" or "blur" or a different masker if preferred.
    # The background color for explanation plots defaults to the average background color
    # of masked regions.
    try:
        # Using 'blur' with a small sigma is often faster and visually clearer than 'inpaint_telea'
        # You can adjust sigma based on image size/complexity.
        # Or try different masker types.
        # masker = shap.maskers.Image("blur", config.INPUT_SHAPE)
        masker = shap.maskers.Image("inpaint_telea", config.INPUT_SHAPE)
    except Exception as e:
        print(f"Error creating SHAP masker: {e}. SHAP explanation skipped.")
        return

    # Create the explainer. The model output should ideally be pre-softmax logits
    # for TreeExplainer or KernelExplainer for better SHAP properties.
    # However, for deep image models and maskers, Explainer with a model that outputs
    # probabilities (like softmax) is commonly used, although it's an approximation.
    # Using the model directly with softmax output.
    try:
        explainer = shap.Explainer(model, masker, output_names=classes)
    except Exception as e:
        print(f"Error creating SHAP explainer: {e}. SHAP explanation skipped.")
        return

    print(
        f"Generating SHAP explanations for {num_samples_per_class} samples per class..."
    )
    print(
        "Note: SHAP explanation for deep image models can be computationally intensive."
    )

    output_dir = os.path.join(
        config.OUTPUT_DIR, f"{model.name.replace('/', '_')}_shap_explanations"
    )
    os.makedirs(output_dir, exist_ok=True)

    explained_count = 0
    # Iterate through each class present in the loaded original data
    for class_name in data_dict.keys():
        print(f"\nExplaining samples for class: {class_name}")
        class_images = data_dict[class_name]

        if not class_images:
            print(f"No original images available for class {class_name}. Skipping.")
            continue

        # Randomly sample images from the original data for this class
        # Ensure we don't sample more images than available
        num_samples_to_explain = min(num_samples_per_class, len(class_images))
        if num_samples_to_explain == 0:
            continue

        sample_indices = np.random.choice(
            len(class_images), num_samples_to_explain, replace=False
        )
        x_explained_orig = np.stack(
            [class_images[i] for i in sample_indices]
        )  # Original images [0, 255] uint8

        # Scale the images the same way the model input was scaled during training
        # Assuming scaling to [0, 1] was applied in data_processor.py
        x_explained_scaled = (
            x_explained_orig.astype(np.float32) / 255.0
        )  # Scale to [0, 1]

        try:
            # Calculate SHAP values. max_evals controls the number of model evaluations.
            # Lower values are faster but less accurate. batch_size helps speed up.
            # outputs=shap.Explanation.argsort.flip[:len(classes)] requests SHAP values
            # for the top 'len(classes)' predicted classes (all classes in this case).
            shap_values = explainer(
                x_explained_scaled,  # Pass scaled data to explainer
                max_evals=512,  # Can adjust max_evals for speed/accuracy
                batch_size=config.BATCH_SIZE,
                outputs=shap.Explanation.argsort.flip[: len(classes)],
            )

            # Plot the explanations
            # Use the original images for plotting as SHAP handles the overlay
            # shap.image_plot expects images in the same format as the model input,
            # but also needs the original image for display. Pass original images [0,255].
            fig = shap.image_plot(
                shap_values,
                x_explained_orig,  # Pass original images [0, 255] for display
                true_labels=[class_name] * num_samples_to_explain,  # List true labels
                show=False,  # Don't auto-show, we'll save
            )

            # Save the figure
            save_path = os.path.join(output_dir, f"{class_name}_shap_samples.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)  # Close the figure to free memory

            explained_count += num_samples_to_explain
            print(
                f"Generated SHAP plots for {num_samples_to_explain} samples of class {class_name}."
            )

        except Exception as e:
            print(
                f"Error generating SHAP values or plotting for class {class_name}: {e}. Skipping this class."
            )
            # Ensure the figure is closed even on error if shap.image_plot was called
            plt.close("all")

    print(
        f"\n--- Finished SHAP Explanation. Total samples explained: {explained_count} ---"
    )


if __name__ == "__main__":
    # Example Usage:
    # Needs a dummy model and dummy data_dict
    from config import cfg
    from models import build_model
    import random

    print("Running example SHAP explanation (dummy model and data)...")
    # Create a simple dummy model
    dummy_model = build_model(
        base_model_name="MobileNet",  # Example model name, adjust as needed
        input_shape=cfg.INPUT_SHAPE,
        num_classes=cfg.NUM_CLASSES,
    )
    dummy_model.compile(
        optimizer="adam", loss="categorical_crossentropy"
    )  # Compile is needed for explainer

    # Create dummy data_dict
    dummy_data_dict = {}
    dummy_img = np.random.randint(0, 256, size=cfg.INPUT_SHAPE, dtype=np.uint8)
    for cls in cfg.CLASSES:
        # Create 10 dummy images per class
        dummy_data_dict[cls] = [
            np.random.randint(0, 256, size=cfg.INPUT_SHAPE, dtype=np.uint8)
            for _ in range(10)
        ]

    # Run SHAP explanation on dummy data
    run_shap_explanation(
        dummy_model, dummy_data_dict, cfg.CLASSES, cfg, num_samples_per_class=1
    )
    print("Dummy SHAP explanation complete.")
