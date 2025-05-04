from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
import os

from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from keras.api.optimizers import Adam

from config import cfg
from utils import plot_acc, plot_loss, save_history
from evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
)


def train_model_phase(
    model, x_train, y_train, x_val, y_val, config, phase_name, initial_threshold=0
):
    """
    Trains the model for a specific phase (transfer learning or fine-tuning).

    Args:
        model (tf.keras.Model): The model to train.
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels (one-hot encoded or mixed).
        x_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels (one-hot encoded).
        config (Config): Configuration object.
        phase_name (str): Name of the training phase ('transfer_learning' or 'fine_tuning').
        initial_threshold (float): Minimum value for the monitored quantity to trigger saving (for ModelCheckpoint).

    Returns:
        tuple: (Keras History object, trained model)
    """
    print(f"\n--- Starting {phase_name} ---")

    epochs = (
        config.TRANSFER_LEARNING_EPOCHS
        if phase_name == "transfer_learning"
        else config.FINE_TUNING_EPOCHS
    )
    learning_rate = (
        config.TRANSFER_LEARNING_LR
        if phase_name == "transfer_learning"
        else config.FINE_TUNING_LR
    )

    # Set trainability for the phase
    if phase_name == "transfer_learning":
        # Freeze the base model
        model.trainable = False
        # Set the classification head layers to trainable
        # The head is typically the layers added after the base_model layer
        # In our build_model, this is everything after base_model(x_input, training=False)
        # We can access these layers by slicing model.layers if the structure is simple,
        # or by explicitly setting them if the head is wrapped in a Sequential model.
        # Assuming the head is directly added after the base_model layer:
        # The base_model layer itself is usually model.layers[-2] (before the output layer)
        # Let's just iterate and check if a layer belongs to the base model
        print("Setting base model layers to non-trainable...")
        for layer in model.layers:
            if layer.name != model.layers[-1].name:  # Exclude the final dense layer
                layer.trainable = False

        # Explicitly set the output layer to trainable (already true, but good practice)
        model.layers[-1].trainable = True

        # Also set layers in the classification head to trainable if use_complex_head was True
        # Find the GlobalAvgPool2D layer
        gap_layer_idx = None
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tf.keras.layers.GlobalAvgPool2D):
                gap_layer_idx = i
                break
        if gap_layer_idx is not None:
            print("Setting classification head layers to trainable...")
            for i in range(gap_layer_idx, len(model.layers)):
                model.layers[i].trainable = True
        else:
            # Simple head (GAP directly to Dense) - only Dense layer needs training
            model.layers[-1].trainable = True

    elif phase_name == "fine_tuning":
        print("Setting the entire model (or parts of it) to trainable...")
        # Unfreeze the whole model or specific layers
        model.trainable = True
        # Optional: Unfreeze only some layers from the base model
        # For example, unfreeze the last few blocks of the base model
        # for layer in model.layers[-50:]: # Example: unfreeze last 50 layers
        #    if not isinstance(layer, (layers.BatchNormalization, layers.LayerNormalization)):
        #       layer.trainable = True
        # pass # model.trainable = True already handles this

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",  # Compatible with MixUp's soft labels
        metrics=["accuracy"],
    )

    # Print trainable parameters count after setting trainability for the phase
    total_params = sum(layer.count_params() for layer in model.layers)
    trainable_params = sum(layer.count_params() for layer in model.layers if layer.trainable)
    non_trainable_params = total_params - trainable_params
    print("Model summary for this phase:")
    print(f"Total params: {total_params/1e6:.2f}M")
    print(f"Trainable params: {trainable_params/1e6:.2f}M")
    print(f"Non-Trainable params: {non_trainable_params/1e6:.2f}M")

    model_name = model.name.replace("/", "_")  # Sanitize name for filenames
    best_weights_path = os.path.join(
        config.OUTPUT_DIR,
        f"{config.dataset_name}_{model_name}_{phase_name}.weights.h5",
    )

    callbacks_list = [
        ModelCheckpoint(
            filepath=best_weights_path,
            monitor="val_accuracy",
            mode="max",
            initial_value_threshold=initial_threshold,  # Only save if better than this threshold
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"Training for {epochs} epochs with learning rate {learning_rate}...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=config.BATCH_SIZE,
        validation_data=(x_val, y_val),
        validation_batch_size=config.BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,  # Show training progress
    )

    print(f"\nFinished {phase_name}.")

    # Load best weights before returning the model
    if os.path.exists(best_weights_path):
        print(f"Loading best weights from {best_weights_path}")
        model.load_weights(best_weights_path)
    else:
        print(
            f"No best weights found at {best_weights_path}. Using final epoch weights."
        )

    return history, model


def run_training_pipeline(
    model, x_train, y_train, x_val, y_val, x_test, y_test, config
):
    """
    Runs the full training pipeline (transfer learning + potential fine-tuning)
     for a given model.

    Args:
        model (tf.keras.Model): The Keras model to train.
        x_train (np.ndarray): Training images.
        y_train (np.ndarray): Training labels (one-hot encoded or mixed).
        x_val (np.ndarray): Validation images.
        y_val (np.ndarray): Validation labels (one-hot encoded).
        x_test (np.ndarray): Test images.
        y_test (np.ndarray): Test labels (one-hot encoded).
        config (Config): Configuration object.

    Returns:
        tuple: (Trained Keras model, numpy array of test predictions)
    """
    model_name = model.name.replace("/", "_")
    dataset_name = "skin-cancer-7-classes"  # Use a consistent name

    best_acc_ph1 = 0.0  # Track best validation accuracy from phase 1

    # --- Phase 1: Transfer Learning (Train Head) ---
    history_ph1, model = train_model_phase(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        config,
        phase_name="transfer_learning",
        initial_threshold=0,  # Always save the first better result
    )

    # Evaluate and plot results for Phase 1
    print(f"\n--- Evaluating {model_name} - Transfer Learning ---")
    loss_ph1, acc_ph1 = evaluate_model(
        model, x_test, y_test
    )  # Returns only loss and accuracy
    print(f"Transfer Learning test scores (loss, acc): [{loss_ph1:.4f}, {acc_ph1:.4f}]")

    y_pred_prob_ph1 = predict_prob(model, x_test, config.BATCH_SIZE)
    y_pred_ph1 = np.argmax(y_pred_prob_ph1, axis=1)
    y_true_test = np.argmax(y_test, axis=1)

    print("\nMetrics for Transfer Learning on Test Set:")
    calculate_metrics(y_true_test, y_pred_ph1)
    cm_ph1 = confusion_matrix(y_true_test, y_pred_ph1)
    plot_confusion_matrix(
        cm_ph1,
        config.CLASSES,
        title=f"Confusion Matrix: {model_name} - Transfer Learning",
        save_path=os.path.join(
            config.OUTPUT_DIR, f"{model_name}_{dataset_name}_cm_ph1.png"
        ),
    )

    plot_acc(
        history_ph1,
        f"{model_name} - Transfer Learning",
        save_path=os.path.join(
            config.OUTPUT_DIR, f"{model_name}_{dataset_name}_acc_ph1.png"
        ),
    )
    plot_loss(
        history_ph1,
        f"{model_name} - Transfer Learning",
        save_path=os.path.join(
            config.OUTPUT_DIR, f"{model_name}_{dataset_name}_loss_ph1.png"
        ),
    )
    save_history(
        history_ph1,
        os.path.join(config.OUTPUT_DIR, f"{model_name}_{dataset_name}_history_ph1.csv"),
    )

    # Get best val_accuracy from phase 1 history to use as threshold for phase 2
    best_acc_ph1 = max(history_ph1.history.get("val_accuracy", [0.0]))
    print(f"\nBest Validation Accuracy from Transfer Learning: {best_acc_ph1:.4f}")

    # --- Phase 2: Fine-Tuning ---
    # Decide if fine-tuning is needed based on a threshold (or always run)
    run_fine_tuning = False
    if (
        best_acc_ph1 < config.FINE_TUNING_ACC_THRESHOLD
        or config.FINE_TUNING_ACC_THRESHOLD >= 1.0
    ):
        # Run fine-tuning if below threshold OR if threshold is set high (>=1.0) meaning always run
        run_fine_tuning = True
        print(
            f"\nValidation accuracy ({best_acc_ph1:.4f}) is below threshold ({config.FINE_TUNING_ACC_THRESHOLD:.4f}). Proceeding to Fine-Tuning."
        )
    else:
        print(
            f"\nValidation accuracy ({best_acc_ph1:.4f}) met or exceeded threshold ({config.FINE_TUNING_ACC_THRESHOLD:.4f}). Skipping Fine-Tuning."
        )
        # If skipping fine-tuning, the final model/predictions are from Phase 1
        final_model = model
        y_pred_prob_final = y_pred_prob_ph1
        y_pred_final = y_pred_ph1

    if run_fine_tuning:
        # Reload the model structure to apply fine-tuning trainability correctly from scratch
        # Or just ensure model.trainable = True is set correctly in train_model_phase
        # Let's ensure train_model_phase handles trainability setting properly
        history_ph2, model = train_model_phase(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            config,
            phase_name="fine_tuning",
            initial_threshold=best_acc_ph1,  # Only save phase 2 weights if they improve over phase 1 best
        )

        # Evaluate and plot results for Phase 2
        print(f"\n--- Evaluating {model_name} - Fine-Tuning ---")
        loss_ph2, acc_ph2 = evaluate_model(
            model, x_test, y_test
        )  # Returns only loss and accuracy
        print(f"Fine-Tuning test scores (loss, acc): [{loss_ph2:.4f}, {acc_ph2:.4f}]")

        y_pred_prob_final = predict_prob(model, x_test, config.BATCH_SIZE)
        y_pred_final = np.argmax(y_pred_prob_final, axis=1)
        y_true_test = np.argmax(y_test, axis=1)

        print("\nMetrics for Fine-Tuning on Test Set:")
        calculate_metrics(y_true_test, y_pred_final)
        cm_ph2 = confusion_matrix(y_true_test, y_pred_final)
        plot_confusion_matrix(
            cm_ph2,
            config.CLASSES,
            title=f"Confusion Matrix: {model_name} - Fine-Tuning",
            save_path=os.path.join(
                config.OUTPUT_DIR, f"{model_name}_{dataset_name}_cm_ph2.png"
            ),
        )

        plot_acc(
            history_ph2,
            f"{model_name} - Fine-Tuning",
            save_path=os.path.join(
                config.OUTPUT_DIR, f"{model_name}_{dataset_name}_acc_ph2.png"
            ),
        )
        plot_loss(
            history_ph2,
            f"{model_name} - Fine-Tuning",
            save_path=os.path.join(
                config.OUTPUT_DIR, f"{model_name}_{dataset_name}_loss_ph2.png"
            ),
        )
        save_history(
            history_ph2,
            os.path.join(
                config.OUTPUT_DIR, f"{model_name}_{dataset_name}_history_ph2.csv"
            ),
        )

        final_model = model

    # Save the final best model after potential fine-tuning
    final_model_path = os.path.join(
        config.OUTPUT_DIR, f"{model_name}_{dataset_name}_final_model.keras"
    )
    print(f"\nSaving final trained model to {final_model_path}")
    final_model.save(final_model_path)

    return final_model, y_pred_prob_final, y_pred_final, y_true_test


# Helper to evaluate and predict (can be in evaluate.py but used heavily in train)
def evaluate_model(model, x, y):
    """Evaluates the model on the given data."""
    print("Evaluating model...")
    scores = model.evaluate(x, y, verbose=1)
    # The evaluate method returns [loss, accuracy, ...]
    # Ensure it returns at least 2 values or handle list length
    if len(scores) >= 2:
        return scores[0], scores[1]  # Return loss and accuracy
    else:
        return scores[0], None  # Return loss, accuracy is None if not available


def predict_prob(model, x, batch_size):
    """Returns class probabilities for the input data."""
    print("Predicting probabilities...")
    return model.predict(x, batch_size=batch_size, verbose=1)


def predict(model, x, batch_size):
    """Returns predicted class indices for the input data."""
    predictions = predict_prob(model, x, batch_size)
    return np.argmax(predictions, axis=1)


# The calculate_metrics and plot_confusion_matrix are moved to evaluate.py
# but imported here because they are used within run_training_pipeline's evaluation steps.

if __name__ == "__main__":
    # Example Usage:
    # This would typically be called from main.py
    # Need dummy data or load real data here for testing
    print("Running example training pipeline (dummy data)...")
    from models import build_model
    from data_processor import split_data, scale_images

    # Create dummy data matching expected shapes
    num_samples = 500
    x_dummy = np.random.randint(
        0, 256, size=(num_samples, cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.uint8
    )
    y_dummy_indices = np.random.randint(0, cfg.NUM_CLASSES, size=num_samples)
    y_dummy_onehot = tf.one_hot(y_dummy_indices, cfg.NUM_CLASSES).numpy()

    x_dummy_scaled = scale_images(x_dummy, scale_range=(0.0, 1.0))

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        x_dummy_scaled, y_dummy_onehot, test_size=0.2, val_size=0.2
    )

    dummy_model = build_model("MobileNet", cfg.INPUT_SHAPE, cfg.NUM_CLASSES)

    # Set a threshold that will likely trigger fine-tuning on dummy data
    cfg.FINE_TUNING_ACC_THRESHOLD = 0.5
    cfg.TRANSFER_LEARNING_EPOCHS = 2
    cfg.FINE_TUNING_EPOCHS = 2  # Short epochs for example

    trained_model, y_pred_prob, y_pred_index, y_true_index = run_training_pipeline(
        dummy_model, x_train, y_train, x_val, y_val, x_test, y_test, cfg
    )

    print("\nDummy training complete.")
