# 🧠 Model Training with MLflow and VGG19

This script facilitates **image classification model training** and **fine-tuning** using **TensorFlow/Keras**, powered by **transfer learning (VGG19)**, with full experiment tracking via **MLflow**.

---
## ⚙️ Features

- Trainable pipeline using pre-trained **VGG19** as base model
- Real-time **data augmentation** with `ImageDataGenerator`
- **Training + Fine-tuning** modes supported
- Integrated with **MLflow** for:
    - Logging parameters, metrics, and models
    - Remote/Local experiment tracking
- UUID-based model versioning for saved models

---

## 🧾 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Common dependencies include:
- `tensorflow`
- `keras`
- `mlflow`
- `PyYAML`
- `matplotlib`
- `argparse`

---

## 🛠️ Configuration (`params.yaml`)

```yaml
model:
  trainable: true
  image_size: [224, 224]
  train_path: "path/to/train"
  test_path: "path/to/test"
  loss: "categorical_crossentropy"
  optimizer: "adam"
  metrics: ["accuracy"]
  epochs: 10
  fine_tune_epochs: 5
  sav_dir: "models/vgg19_model.h5"
  name: "vgg19_transfer"

img_augment:
  rescale: 0.00392156862745098
  shear_range: 0.2
  zoom_range: 0.2
  horizontal_flip: true
  vertical_flip: false
  brightness_range: [0.5, 1.5]
  class_mode: "categorical"
  batch_size: 32

mlflow_config:
  remote_server_uri: "http://localhost:5000"
  experiment_name: "Image_Classification"
  registered_model_name: "VGG19_Image_Classifier"

load_data:
  num_classes: 10

run_config:
  mode: "train"  # or "fine_tune"
```

---

## 🚀 How to Run

### 🏋️‍♂️ Train a New Model

```bash
python model_train_mlflow.py --config params.yaml
```

### 🔧 Fine-tune an Existing Model

Update `run_config: mode: fine_tune` in `params.yaml` and run:

```bash
python model_train_mlflow.py --config params.yaml
```

---

## 🧪 MLflow Tracking

This script logs:
- **Params**: epochs, loss, optimizer, etc.
- **Metrics**: accuracy, validation loss, etc.
- **Model Artifacts**: Saved Keras models

If a remote URI is configured, the model is also **registered to MLflow**.

---

## 🧼 Utility Functions

### `add_uuid_to_filename(filename)`
Adds a UUID suffix to your saved model file.

### `add_uuid_and_suffix_to_filename(filename, suffix)`
Adds a UUID and a custom suffix like `"finetuned"` to differentiate versions.

---

## 📈 Outputs

- Logs and metrics in MLflow UI
- Trained and fine-tuned models saved with UUID
- Console summary of model architecture and training logs

---

## 📝 Notes

- Ensure your data is structured for `ImageDataGenerator.flow_from_directory`:
  ```
  train/
    class_1/
    class_2/
    ...
  test/
    class_1/
    class_2/
    ...
  ```
- The script assumes the presence of helper files `get_data.py` and `model_train.py`.
---