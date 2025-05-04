import tensorflow as tf
from keras import layers, Model
from keras.api.applications import ResNet50, DenseNet169, Xception, MobileNet, EfficientNetB0, EfficientNetB4

from config import cfg

def build_classification_head(input_tensor, num_classes, use_complex_head=False):
    """
    Builds the classification head for a pre-trained base model.

    Args:
        input_tensor (tf.Tensor): The output tensor from the base model.
        num_classes (int): The number of output classes.
        use_complex_head (bool): If True, use the more complex head from MobileNet example.

    Returns:
        tf.Tensor: The output tensor of the classification head.
    """
    x = layers.GlobalAvgPool2D()(input_tensor)

    if use_complex_head:
        # Head from MobileNet example in the notebook
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)

    output = layers.Dense(num_classes, activation='softmax')(x)
    return output

def build_model(base_model_name, input_shape, num_classes):
    """
    Builds a transfer learning model with a specified pre-trained base.

    Args:
        base_model_name (str): Name of the pre-trained model (e.g., 'ResNet50').
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes.

    Returns:
        Model: The built Keras model.
    """
    # Define a dictionary mapping base model names to their classes and head type
    base_models = {
        'ResNet50': (ResNet50, False),
        'DenseNet169': (DenseNet169, False),
        'Xception': (Xception, False),
        'MobileNet': (MobileNet, True), # MobileNet in notebook used complex head
        'EfficientNetB0': (EfficientNetB0, False),
        'EfficientNetB4': (EfficientNetB4, True), # EfficientNetB4 in notebook used complex head
    }

    if base_model_name not in base_models:
        raise ValueError(f"Unknown base model: {base_model_name}. Choose from {list(base_models.keys())}")

    BaseModelClass, use_complex_head = base_models[base_model_name]

    print(f"\nBuilding model with base: {base_model_name}")

    # Load pre-trained base model, exclude top classification layer
    base_model = BaseModelClass(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )
    # Initially freeze the base model for transfer learning phase
    base_model.trainable = False

    # Build the full model
    x_input = tf.keras.Input(shape=input_shape)
    # Keras-CV MixUp uses Float32, ensure input layer matches
    # Although the scaling step handles conversion, being explicit here is good.
    # x_input = tf.cast(x_input, tf.float32)

    # Add a preprocessing layer if needed, e.g., scaling or specific model preprocessing
    # base_model.preprocess_input could be added here for specific models,
    # but simple [0,1] scaling is handled in data_processor for now.
    # x = base_model.preprocess_input(x_input) if hasattr(base_model, 'preprocess_input') else x_input
    # Let's assume scaling is handled before this.

    x = base_model(x_input, training=False) # Ensure base model runs in inference mode when frozen

    # Add classification head
    output = build_classification_head(x, num_classes, use_complex_head=use_complex_head)

    model = Model(inputs=x_input, outputs=output, name=base_model_name)

    print(f"{model.name}: {len(model.layers)} layers.")
    total_params = sum(layer.count_params() for layer in model.layers)
    trainable_params = sum(layer.count_params() for layer in model.layers if layer.trainable)
    non_trainable_params = total_params - trainable_params

    print(f"Total params: {total_params/1e6:.2f}M")
    print(f"Trainable params (initially frozen): {trainable_params/1e6:.2f}M")
    print(f"Non-Trainable params (initially frozen): {non_trainable_params/1e6:.2f}M")


    return model

if __name__ == '__main__':
    # Example usage:
    resnet_model = build_model('ResNet50', cfg.INPUT_SHAPE, cfg.NUM_CLASSES)
    mobilenet_model = build_model('MobileNet', cfg.INPUT_SHAPE, cfg.NUM_CLASSES)
    # You can print model summaries if desired
    # resnet_model.summary()
    # mobilenet_model.summary()