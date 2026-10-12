# 🧠 Brain Tumor MRI Image Classification

A deep-learning project that classifies brain MRI scans into four categories — **Glioma, Meningioma, Pituitary tumor, and No Tumor** — using a **custom CNN built from scratch** and **transfer learning** (ResNet50, MobileNetV2, EfficientNetB0). The best model is deployed in an interactive **Streamlit** web app that predicts the tumor type and shows a confidence score for any uploaded MRI image.

> ⚠️ **Disclaimer:** This project is for education and research only. It is a decision-support demo and **not** a medical device or a diagnosis. Always consult a qualified medical professional.

🌐 **Live App:** _add your Streamlit link here_
📓 **Notebook:** [`Brain_Tumor_MRI_Classification.ipynb`](Brain_Tumor_MRI_Classification.ipynb)

---

## 📌 Table of Contents
- [Problem Statement](#-problem-statement)
- [Real-World Use Cases](#-real-world-use-cases)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Models](#-models)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Streamlit App](#-streamlit-app)
- [Tech Stack](#-tech-stack)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🎯 Problem Statement

Identifying the type of a brain tumor from an MRI scan is the first step in choosing a treatment, but manual review is time-consuming and depends on the radiologist's experience. This project builds a deep-learning classifier that:

1. Classifies MRI images into **4 classes** (Glioma, Meningioma, Pituitary, No Tumor).
2. Compares a **custom CNN** against **pretrained transfer-learning models**.
3. Deploys the best model in a **user-friendly web app** for real-time predictions.

## 💡 Real-World Use Cases

| Use case | Description |
|---|---|
| **AI-assisted diagnosis** | Give radiologists a fast first opinion on the tumor type. |
| **Early detection & triage** | Flag high-risk scans for urgent specialist review. |
| **Research & clinical trials** | Group patient scans by tumor type to build study cohorts. |
| **Second-opinion systems** | Support telemedicine in under-resourced regions. |

---

## 📂 Dataset

**Brain Tumor MRI Multi-Class Dataset** (train / valid / test folders, each with its images and a `_classes.csv` label file).

| Split | Images |
|---|---:|
| Train | 1,695 |
| Validation | 502 |
| Test | 246 |
| **Total** | **2,443** |

| Class | Images | Share |
|---|---:|---:|
| Glioma | 805 | ~33% |
| Pituitary | 610 | ~25% |
| Meningioma | 545 | ~22% |
| No Tumor | 483 | ~20% |

- All images are **640 × 640 RGB**.
- Classes are **mildly imbalanced** (largest / smallest ratio ≈ 1.67), so **class weights** are used during training.
- The dataset is **not included** in this repository. Download it from the source below and place it as shown in [Installation & Usage](#-installation--usage).

📎 **Dataset link:** _add the dataset link here_

---

## 🔄 Project Workflow

1. **Know the data** – structure, class balance, resolution, missing values, duplicates.
2. **Data wrangling** – build a clean table of images and labels, verify every file exists.
3. **Data visualization** – class distribution, sample MRIs, pixel-intensity analysis, mean image per class.
4. **Hypothesis testing** – ANOVA on pixel intensity across classes; chi-square on class proportions across splits.
5. **Pre-processing** – resize to **224 × 224**, normalize pixels to **0–1**.
6. **Data augmentation** (training only) – rotation, horizontal/vertical flip, zoom, shifts, brightness.
7. **Model building** – custom CNN and three transfer-learning models.
8. **Training** – `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, class weights.
9. **Fine-tuning** – unfreeze the top layers of the best pretrained model with a very small learning rate.
10. **Evaluation** – accuracy, precision, recall, F1-score, confusion matrix, accuracy/loss curves.
11. **Model comparison** – accuracy, parameters, and prediction speed.
12. **Deployment** – best model saved as `.h5` and served through Streamlit.

---

## 🤖 Models

| # | Model | Type | Notes |
|---|---|---|---|
| 1 | **Custom CNN** | From scratch | 4 conv blocks (Conv + BatchNorm + MaxPool + Dropout), GlobalAveragePooling, Dense head |
| 2 | **ResNet50** | Transfer learning | ImageNet weights, residual connections |
| 3 | **MobileNetV2** | Transfer learning | Light-weight, fast predictions |
| 4 | **EfficientNetB0** | Transfer learning | High accuracy with few parameters |
| 5 | **Best pretrained model – fine-tuned** | Fine-tuning | Top 30 layers unfrozen, learning rate 1e-5 |

All transfer-learning models share the same classification head (GlobalAveragePooling → Dense(256) → BatchNorm → Dropout → Softmax) so the comparison is fair. A small input-conversion layer maps the 0–1 images to the range each pretrained network expects.

---

## 📊 Results

_Fill this table with the output of the **Model Comparison** section of the notebook (also saved as `models/model_comparison.csv`)._

| Model | Test Accuracy | Precision | Recall | F1-score | Params (M) | Inference (ms/img) |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | – | – | – | – | – | – |
| ResNet50 | – | – | – | – | – | – |
| MobileNetV2 | – | – | – | – | – | – |
| EfficientNetB0 | – | – | – | – | – | – |
| Best model (fine-tuned) | – | – | – | – | – | – |

**Selected model:** _name_ — chosen for the highest F1-score on the held-out test set.

_Optional: add screenshots of the confusion matrix and training curves here._

```
![Confusion Matrix](images/confusion_matrix.png)
![Training Curves](images/training_curves.png)
```

---

## 🗂 Project Structure

```
brain-tumor-mri-classification/
│── Brain_Tumor_MRI_Classification.ipynb   # data analysis, training, evaluation, model saving
│── app.py                                 # Streamlit web application
│── requirements.txt                       # Python dependencies
│── README.md
│── models/
│   ├── best_model.h5                      # best model used by the app
│   ├── class_names.json                   # class order used for predictions
│   ├── model_comparison.csv               # comparison table
│   └── (other .h5 models, optional)
└── images/                                # screenshots for the README (optional)
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (Google Colab recommended)
1. Open `Brain_Tumor_MRI_Classification.ipynb` in **Google Colab** and set the runtime to **GPU** (*Runtime → Change runtime type → T4 GPU*).
2. Upload the dataset zip file(s) to the Colab **Files** panel and wait for the upload to finish.
3. Run all cells in order. The notebook extracts the zip, builds the dataset table, trains all models, compares them, and saves everything to `/content/models`.
4. Download the generated `models.zip` and unzip it into the `models/` folder of this repository.

### 4. Run the web app
```bash
streamlit run app.py
```
Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## 🖥 Streamlit App

**Features**
- Upload a JPG / PNG brain MRI image.
- See the **predicted tumor type** and the **confidence score**.
- Bar chart with the **probability of every class**.
- Short description of the predicted class.
- Model selector in the sidebar (every `.h5` file in `models/`).
- Low-confidence warning (below 60%) recommending review by a radiologist.

**Pre-processing used by the app** (identical to training): convert to RGB → resize to 224 × 224 → divide pixels by 255.

📸 _Add a screenshot of the app here: `![App](images/app_screenshot.png)`_

### Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `models/best_model.h5` + `models/class_names.json` to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your repository, and select `app.py` as the main file.

> **Note on file size:** GitHub blocks files larger than **100 MB**. If a `.h5` file is bigger than that (ResNet50 models can be), upload only the smaller best model, or use [Git LFS](https://git-lfs.com/).

---

## 🛠 Tech Stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `SciPy` · `Pillow` · `Streamlit`

---

## ⚠️ Limitations & Future Work

- The model is trained on one public dataset; it needs validation on data from **other hospitals and MRI scanners** before any clinical use.
- Some tumor classes can look similar on MRI, so confusion between classes is possible — always check the confidence score.
- **Future work:** Grad-CAM heat-maps to show *where* the model looks, model ensembling, larger multi-center data, and clinical validation.

---

## 👤 Author

**Your Name** · Roshan Chhotulal Patil

## 📄 License

This project is for educational purposes. Add a license of your choice (for example MIT) and check the license of the dataset you use.
