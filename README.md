# Heart Disease Prediction - Machine Learning Project

A comprehensive machine learning project for predicting heart disease using multiple classification algorithms with extensive analysis and visualizations.

## 📊 Project Overview

This project implements and compares **5 different machine learning models** to predict the presence of heart disease in patients. The project includes comprehensive exploratory data analysis, model training, evaluation, and visualization.

## 🎯 Features

- **5 Machine Learning Models:**
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)

- **Comprehensive Analysis:**
  - Train / Validation / Test split (70 / 15 / 15)
  - 5-Fold Cross-Validation
  - Overfitting detection via train-test gap analysis
  - ROC Curve Analysis
  - Feature Importance Analysis
  - Confusion Matrix for each model

- **Rich Visualizations:**
  - Target Distribution Analysis
  - Feature Correlation Heatmaps
  - Age Distribution by Disease Status
  - Categorical Features Analysis
  - Model Performance Comparison Charts
  - ROC Curves with AUC Scores
  - Learning Curves for Overfitting Detection
  - Cross-Validation Box Plots

## 📁 Dataset

The dataset (`heart.csv`) is sourced from the UCI Heart Disease dataset. After removing duplicate rows, it contains **1,025 patient records** across 13 input features plus the target label:

| Feature  | Description                                    |
| -------- | ----------------------------------------------- |
| age      | Age of the patient                              |
| sex      | Sex (1 = male, 0 = female)                      |
| cp       | Chest pain type (0–3)                           |
| trestbps | Resting blood pressure (mm Hg)                  |
| chol     | Serum cholesterol (mg/dl)                       |
| fbs      | Fasting blood sugar > 120 mg/dl                 |
| restecg  | Resting electrocardiographic results            |
| thalach  | Maximum heart rate achieved                     |
| exang    | Exercise induced angina                         |
| oldpeak  | ST depression induced by exercise               |
| slope    | Slope of the peak exercise ST segment           |
| ca       | Number of major vessels colored by fluoroscopy  |
| thal     | Thalassemia                                     |
| target   | Heart disease (1 = disease, 0 = no disease)     |

Split used for the final results below: **717 train / 154 validation / 154 test.**

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

### Running the Project

**Option 1: Local execution**
```bash
git clone https://github.com/ahmedgalalxxx/Heart-Disease-Prediction---ML.git
cd Heart-Disease-Prediction---ML
jupyter notebook heart_disease_ml_project.ipynb
```

**Option 2: Google Colab**
1. Upload `heart_disease_ml_project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload the `heart.csv` file when prompted
3. Run all cells

## 📈 Model Performance

All 5 models were evaluated on a held-out test set (154 samples) after 5-fold cross-validation on the training set:

| Model               | Train Acc. | Val Acc. | Test Acc.  | CV Score | Precision | Recall  | F1     | Overfitting Gap |
|---------------------|-----------:|---------:|-----------:|---------:|----------:|--------:|-------:|----------------:|
| **Random Forest**   | 0.9958     | 0.9870   | **0.9675** | 0.9443   | 0.9744    | 0.9620  | **0.9682** | 0.0283       |
| Decision Tree       | 0.9344     | 0.8571   | 0.8896     | 0.8955   | 0.8523    | 0.9494  | 0.8982 | 0.0448           |
| SVM                 | 0.9358     | 0.9221   | 0.8766     | 0.8856   | 0.8750    | 0.8861  | 0.8805 | 0.0592           |
| KNN                 | 0.8954     | 0.8701   | 0.8571     | 0.8619   | 0.8800    | 0.8354  | 0.8571 | 0.0383           |
| Logistic Regression | 0.8647     | 0.8377   | 0.7987     | 0.8452   | 0.7553    | 0.8987  | 0.8208 | 0.0660           |

**Best model: Random Forest — 96.75% test accuracy, 96.82% F1-score, with only a 2.83% train-test gap** (low overfitting relative to the other models tested).

Confusion matrix (Random Forest, test set):

| | Predicted: No Disease | Predicted: Disease |
|---|---:|---:|
| **Actual: No Disease** | 73 (TN) | 2 (FP) |
| **Actual: Disease** | 3 (FN) | 76 (TP) |

Full classification report:

```
              precision    recall  f1-score   support
No Disease       0.96      0.97      0.97        75
Disease          0.97      0.96      0.97        79
accuracy                              0.97       154
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn

## 📝 Project Structure

```
Heart-Disease-Prediction---ML/
├── heart.csv                       # Dataset
├── heart_disease_ml_project.ipynb  # Main Jupyter notebook
├── README.md                       # Project documentation
└── .gitignore
```

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing and feature scaling
- Multiple classification algorithm implementation
- Model evaluation and comparison techniques
- Overfitting detection and mitigation
- Data visualization best practices
- Cross-validation methodology
- Performance metrics interpretation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Ahmed Elgebaly** — [GitHub](https://github.com/ahmedgalalxxx)

## 🙏 Acknowledgments

- Dataset sourced from the UCI Machine Learning Repository
- Built using scikit-learn and standard ML best practices

---

**Note:** This is an educational project and is not intended for actual medical diagnosis.
