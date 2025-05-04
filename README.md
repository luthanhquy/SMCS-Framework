# ⚕️ SMSC: A Framework with Advanced Sampling Methods for Skin Cancer Classification

Inmodernsociety, environmental pollution and climate change
are considered the main problems affecting to increase in cancer cases.
One of these, skin cancer occurs when there is an overgrowth of abnor
mal cells in the skin. Additionally, Skin cancer can develop in areas faced
with UV (Ultraviolet) radiation. But it can also form in areas that rarely
see the light. Thus, many experiments in both medicine and computer
areas were realized to diagnose and treat this illness. This study intro
duces SMSC (Sampling in MobileNet for Skin Classification), a frame
work that leverages a fine-tuned MobileNet model and advanced sam
pling techniques to address class imbalance in the HAM10000 dataset.
SMSCachieved remarkable results, with a validation accuracy of 96.93%
and a test accuracy of 95.61% for classifying seven skin cancer types. Ad
ditionally, for binary classification between benign and malignant lesions,
the model reached an average validation accuracy of 99.41% and a test
accuracy of 98.92%. SHapley Additive exPlanations (SHAP) was em
ployed to provide interpretability by explaining the model’s decisions at
the pixel level.

## Work Published in following article

Tran, T.V., Nguyen, T.M., Lu, Q.T. (2025). MSC: A Framework with Advanced Sampling Methods for Skin Cancer Classification. In: Zhang, Y., Zhang, LJ. (eds) Web Services – ICWS 2024. ICWS 2024. Lecture Notes in Computer Science, vol 15428. Springer, Cham. https://doi.org/10.1007/978-3-031-77072-2_9

## ✨ Features

- **Data Loading & Preprocessing:** Reads metadata, loads and preprocesses images (resizing, color conversion).
- **Class Balancing:** Employs MixUp data augmentation to generate synthetic images and balance the dataset across all classes.
- **Data Splitting:** Splits the balanced dataset into training, validation, and test sets with stratification.
- **Transfer Learning:** Utilizes several popular pre-trained CNN architectures (ResNet50, DenseNet169, Xception, MobileNet, EfficientNetB0, EfficientNetB4) initialized with ImageNet weights.
- **Two-Phase Training:** Implements a training strategy involving an initial transfer learning phase (training only the classification head) followed by an optional fine-tuning phase (unfreezing and training the entire model at a lower learning rate).
- **Model Evaluation:** Evaluates trained models on the test set using metrics like accuracy, precision, recall, F1 score, and confusion matrices.
- **Model Persistence:** Saves the best performing model weights and the final trained model for later use.
- **Training History:** Records and saves detailed training history (loss, accuracy per epoch).
- **Explainability:** Uses the SHAP library to provide visual explanations of model predictions, highlighting pixels that contribute positively or negatively to a class prediction.
- **Prediction Visualization:** Visualizes sample test set predictions with predicted probabilities for each class.
- **Exploratory Analysis:** Includes an optional step demonstrating K-Means clustering for image segmentation.

## 📊 Dataset

The project uses the [The HAM10000 dataset](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset). It is a collection of 10000+ dermatoscopic images of common pigmented skin lesions.

The dataset includes the following 7 diagnostic categories (classes):

- `akiec`: Actinic keratoses and intraepithelial carcinoma / Bowen's disease
- `bcc`: Basal cell carcinoma
- `bkl`: Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)
- `df`: Dermatofibroma
- `nv`: Melanocytic nevi
- `mel`: Melanoma
- `vasc`: Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)

_Note: The data loading step in `data_loader.py` limits the initial number of original images loaded per class to `MAX_ORIGINAL_PER_CLASS` (default 500), and `data_processor.py` then uses MixUp to augment classes up to `TARGET_TOTAL_PER_CLASS` (default 1000)._

## 📁 Project Structure

The project is organized into several Python files for modularity and clarity:

```
.
├── models              # Contains ours pre-trained proposed models
├── data                # Contains images and csv file
├── config.py           # Configuration settings (paths, hyperparameters, model selection)
├── data_loader.py      # Handles loading original images and metadata
├── data_processor.py   # Implements data balancing (MixUp), scaling, and train/val/test splitting
├── models.py           # Defines the CNN model architectures using pre-trained bases
├── train.py            # Contains the main training pipeline logic (transfer learning & fine-tuning)
├── evaluate.py         # Provides functions for calculating and printing evaluation metrics and reports
├── explain.py          # Implements SHAP for model explainability
├── utils.py            # Utility functions for plotting, visualization, and KMeans clustering
├── main.py             # The main script to orchestrate the entire process
└── requirements.txt    # Lists project dependencies
```

## 🚀 Setup and Installation

1.  **Clone the Repository:**

    ```bash
    # If hosted on Git
    # git clone <repository_url>
    # cd <repository_name>
    ```

    _(If not hosted, just place the Python files and requirements.txt in a directory)._

2.  **Download the Dataset:** Download the HAM10000 dataset from Kaggle or another source. You will need the metadata CSV (`HAM10000_metadata.csv`) and the image files (typically in subdirectories like `Skin Cancer/`).

3.  **Update Configuration:** Open `config.py` and **update the `METADATA_PATH` and `IMAGE_DIR_PATH` variables** to point to the locations where you saved the dataset files on your local machine. Review other parameters like `TARGET_TOTAL_PER_CLASS`, `MODELS_TO_TRAIN`, etc., and adjust if needed.

4.  **Install Dependencies:** Ensure you have Python installed (3.10+ recommended). Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    _Note: Training deep learning models, especially large CNNs, is computationally intensive. A GPU and CUDA setup are highly recommended for reasonable training times. Ensure your TensorFlow installation is compatible with your GPU if you have one._

## ⚙️ Configuration

The `config.py` file contains all the main parameters to customize the project's behavior. Key settings include:

- `METADATA_PATH`, `IMAGE_DIR_PATH`: Paths to your dataset. **(Must Update)**
- `OUTPUT_DIR`: Directory to save results.
- `IMG_SIZE`: Image dimensions for model input.
- `CLASSES`: List of class names (determines the order of output predictions).
- `MAX_ORIGINAL_PER_CLASS`, `TARGET_TOTAL_PER_CLASS`, `MIX_PER_BATCH`: Parameters for data loading and MixUp augmentation.
- `BATCH_SIZE`: Training batch size.
- `TRANSFER_LEARNING_EPOCHS`, `FINE_TUNING_EPOCHS`: Number of epochs for each training phase.
- `TRANSFER_LEARNING_LR`, `FINE_TUNING_LR`: Learning rates for each phase.
- `EARLY_STOPPING_PATIENCE`: Patience for the early stopping callback.
- `FINE_TUNING_ACC_THRESHOLD`: Validation accuracy threshold in Phase 1 to trigger Phase 2 (Fine-Tuning). Set to `0.0` to always run fine-tuning if Phase 1 accuracy is less than 100%.
- `MODELS_TO_TRAIN`: A list of model names (corresponding to functions in `models.py`) that you want to train.
- `SHAP_NUM_SAMPLES_PER_CLASS`: Number of samples per class to explain using SHAP.

## 🏃 Usage

To run the entire classification pipeline (data loading, processing, training selected models, evaluation, SHAP explanation, and prediction visualization), execute the `main.py` script:

```bash
python main.py
```

The script will print progress updates in your console. Training can take a significant amount of time depending on your hardware and the number of models selected in `config.py`.

## 📁 Output

All generated files (trained models, history CSVs, plots) will be saved in the directory specified by `config.OUTPUT_DIR` (default: `./output`).

You can expect files such as:

- `./output/<model_name>_skin-cancer-7-classes_ph1.weights.keras` (Best weights from Phase 1)
- `./output/<model_name>_skin-cancer-7-classes_ph2.weights.keras` (Best weights from Phase 2)
- `./output/<model_name>_skin-cancer-7-classes_final_model.keras` (Final trained model)
- `./output/<model_name>_skin-cancer-7-classes_history_ph1.csv` (Training history for Phase 1)
- `./output/<model_name>_skin-cancer-7-classes_history_ph2.csv` (Training history for Phase 2)
- `./output/<model_name>_skin-cancer-7-classes_cm_ph1.png` (Confusion matrix for Phase 1 test results)
- `./output/<model_name>_skin-cancer-7-classes_cm_ph2.png` (Confusion matrix for Phase 2 test results)
- `./output/<model_name>_skin-cancer-7-classes_acc_ph1.png` (Accuracy plot for Phase 1)
- `./output/<model_name>_skin-cancer-7-classes_loss_ph1.png` (Loss plot for Phase 1)
- `./output/<model_name>_skin-cancer-7-classes_acc_ph2.png` (Accuracy plot for Phase 2)
- `./output/<model_name>_skin-cancer-7-classes_loss_ph2.png` (Loss plot for Phase 2)
- `./output/<model_name>_shap_explanations/<class_name>_shap_samples.png` (SHAP explanation plots)

Console output will include evaluation metrics and the classification report for the test set after each relevant training phase.

## 🤔 Explainability (SHAP)

The project includes a step using SHAP (SHapley Additive exPlanations) to understand which parts of an input image are most influential in the model's prediction for a given class. This helps in interpreting the model's decision-making process. The SHAP explanation plots are saved in the output directory.

