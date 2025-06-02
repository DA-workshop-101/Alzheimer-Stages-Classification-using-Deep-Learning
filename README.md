# 🧠 Alzheimer Stages Classification Using Deep Learning

A full-stack MLOps-enabled deep learning project to classify MRI brain scans into stages of Alzheimer's disease using a VGG19-based model. The system includes an inference API (FastAPI), web frontend, Grad-CAM visualizations, and CI/CD-powered deployment on Google Cloud Run.

***

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [MLOps Workflow](#mlops-workflow)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

***

<h2 id="overview">🧠 Overview</h2>

This project uses a transfer learning approach with VGG19 to classify MRI images into four Alzheimer’s disease stages:

- Non-Demented  
- Very Mild Demented  
- Mild Demented  
- Moderate Demented  

Users upload an image through the frontend. The backend (FastAPI) returns:

- The predicted class  
- Confidence score  
- Grad-CAM heatmap (base64 encoded)

***

<h2 id="project-structure">🗂️ Project Structure</h2>

```
.  
├── webapp/
│   ├── backend/           # FastAPI app and ML logic
│   └── frontend/          # HTML-CSS-JS frontend
├── src/                   # Model training, data preprocessing
├── models/                # Saved models (.h5)
├── reports/               # Training plots, confusion matrix, metrics
├── Dockerfile             # Container config
├── .github/workflows/     # CI/CD workflows
├── requirements.txt       # Development dependencies
├── requirements_prod.txt  # Production (API) dependencies
├── params.yaml            # Configurations (batch size, epochs, paths, etc.)
└── README.md
```

***

<h2 id="key-features">✨ Key Features</h2>

- ✅ Deep learning with VGG19 and custom classification layers  
- ✅ Grad-CAM visualization during inference  
- ✅ FastAPI backend with robust API endpoints (`/predict`, `/ping`)  
- ✅ Fully containerized with Docker  
- ✅ MLflow for model tracking and fine-tuning  
- ✅ CI/CD with GitHub Actions  
- ✅ Production-ready deployment on GCP Cloud Run  
- ✅ Clean HTML-CSS-JS frontend with fetch-based API calls  

***

<h2 id="getting-started">⚙️ Getting Started</h2>

### 1️⃣ Development Environment Setup

For training, experimentation, and visualization:

```bash 
conda create -n alzheimer39 python=3.9  
conda activate alzheimer39  
pip install tensorflow==2.10.*  
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0  
pip install -r requirements.txt  
```

> 💡 GPU support is optional but recommended. You can skip CUDA/CuDNN if using CPU only.


### 2️⃣ Production Environment Setup (Deployment Backend)

This environment is for running the FastAPI-based backend in production (no training):

```bash 
conda create -n alzheimers-api python=3.9  
conda activate alzheimers-api  
pip install -r requirements_prod.txt  
```

> ⚙️ This environment is used in Docker and Cloud Run deployment. It contains only FastAPI, TensorFlow, and essential runtime dependencies.


### 3️⃣ Run FastAPI Locally (for testing backend)

```bash
uvicorn webapp.backend.main:app --reload  
```

App will be accessible at:  
`http://127.0.0.1:8000`  
Docs auto-generated at:  
`http://127.0.0.1:8000/docs`


***

<h2 id="mlops-workflow">🔄 MLOps Workflow</h2>

### 🔬 Training & Evaluation

- Uses `model_train.py` and `model_train_mlflow.py`  
- Model saved in `models/`  
- Training plots saved to `reports/`  
- Test evaluation in `evaluate_test_set.py`  

### 📦 Version Control & Experiments

- MLflow for experiment tracking and fine-tuning  
- Parameters controlled via `params.yaml`  

### 📁 Data Handling

- Dataset split into `train/` and `test/` using `split.py`  
- Classes: `AD`, `CN`, `EMCI`, `LMCI`  

***

<h2 id="model-performance">📊 Model Performance</h2>

Evaluation on the held-out test set showed:

- **Accuracy**: ~92%  
- **F1-Score**: ~90.5%  
- **Avg Confidence**: 87–93% range  
- Confusion Matrix and Classification Report in `reports/`  

***

<h2 id="deployment">🚀 Deployment</h2>

### 🐳 Dockerization

- Backend containerized via a Dockerfile  
- Exposes port 8000 (Cloud Run remaps to 8080)  

```Dockerfile 
EXPOSE 8000  
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]  
 ```



### 🤖 CI/CD – GitHub Actions

- On `main` branch push:  
  - Lint + test  
  - Build Docker image  
  - Push to DockerHub  
  - Deploy to GCP Cloud Run  

### ☁️ Cloud Run Deployment

- Region: `us-central1`  
- Auto-scales between 0–1 instances  
- Secured with service account + GitHub secrets  

### 🌐 Frontend Deployment

- Deployed via Netlify or GitHub Pages  
- Sends fetch requests to Cloud Run backend  

***

<h2 id="contributing">🤝 Contributing</h2>

1. Fork this repo  
2. Create a feature branch (`git checkout -b feature-X`)  
3. Commit your changes  
4. Push and open a PR  

***

<h2 id="license">📄 License</h2>

Licensed under the MIT License.


