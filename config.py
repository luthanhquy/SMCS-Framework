import os

class Config:

    dataset_name = 'HAM10000'  # Dataset name
    # Data Paths
    METADATA_PATH = 'data/HAM10000_metadata.csv' # <<< !!! Update this path locally !!!
    IMAGE_DIR_PATH = 'data/Skin_Cancer' # <<< !!! Update this path locally !!!
    OUTPUT_DIR = 'output' # Directory for saving models, history, plots

    # Image Parameters
    IMG_SIZE = 224
    INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    # Classes order from original notebook (bkl, nv, df, mel, vasc, bcc, akiec)
    CLASSES = [
        'bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec'
    ]
    NUM_CLASSES = len(CLASSES)


    # Data Loading & Balancing Parameters
    # MAX_ORIGINAL_PER_CLASS = 500 # Limit original images per class
    MAX_ORIGINAL_PER_CLASS = 100 # For testing, limit to 100 images per class
    # TARGET_TOTAL_PER_CLASS = 1000 # Target number of images per class after MixUp (sum of original + mixed)
    TARGET_TOTAL_PER_CLASS = 200 # For testing, limit to 200 images per class
    MIX_PER_BATCH = 42 # Batch size for MixUp generation

    # Training Parameters
    BATCH_SIZE = 32
    TRANSFER_LEARNING_EPOCHS = 2
    FINE_TUNING_EPOCHS = 2
    TRANSFER_LEARNING_LR = 1e-3
    FINE_TUNING_LR = 1e-5
    EARLY_STOPPING_PATIENCE = 15
    # Threshold for triggering fine-tuning (e.g., 0.7 for 70% accuracy)
    # Set to 0 to always perform fine-tuning if phase 1 val_acc is not 100%
    # Set to 1.0 to always perform fine-tuning (as 0.99 was unrealistic)
    FINE_TUNING_ACC_THRESHOLD = 0.0 # Trigger fine-tuning if val_acc < this

    # Model Parameters
    # List of models to train. Add/remove model names defined in models.py
    MODELS_TO_TRAIN = ['MobileNet'] # Example: only MobileNet
    # MODELS_TO_TRAIN = ['ResNet50', 'DenseNet169', 'Xception', 'MobileNet', 'EfficientNetB0', 'EfficientNetB4'] # Train all

    # SHAP Explanation Parameters
    SHAP_NUM_SAMPLES_PER_CLASS = 2 # Number of samples to explain per class

    def __init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

# Instantiate the config
cfg = Config()