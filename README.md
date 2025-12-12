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
  - Train/Validation/Test Split (70-15-15)
  - 5-Fold Cross-Validation
  - Overfitting Detection with Learning Curves
  - ROC Curve Analysis
  - Feature Importance Analysis
  - Confusion Matrix for Each Model

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

The dataset contains **1027 patient records** with 14 features:

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia |
| target | Heart disease (1 = disease, 0 = no disease) |

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

### Running the Project

#### Option 1: Local Execution
1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

2. Open the Jupyter notebook
```bash
jupyter notebook heart_disease_prediction.ipynb
```

3. Run all cells sequentially

#### Option 2: Google Colab
1. Upload `heart_disease_prediction.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload the `heart.csv` file when prompted
3. Run all cells

## 📈 Model Performance

The models achieve realistic accuracy scores (85-90%) suitable for a student project:

- **Best Model**: Varies based on data split, typically Random Forest or Logistic Regression
- **Metrics Evaluated**: Accuracy, Precision, Recall, F1-Score, AUC
- **Validation**: 5-Fold Cross-Validation implemented
- **Overfitting Check**: Learning curves and train/test gap analysis

## 📊 Results

The notebook generates:
- Model comparison tables
- Performance metrics visualization
- Confusion matrices for all models
- ROC curves with AUC scores
- Feature importance rankings
- Comprehensive final summary report

Results are automatically saved to `model_results.csv`

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn

## 📝 Project Structure

```
heart-disease-prediction/
│
├── heart.csv                           # Dataset
├── heart_disease_prediction.ipynb      # Main Jupyter notebook
├── heart_disease_ml_project.ipynb      # Alternative compact version
├── README.md                           # Project documentation
└── model_results.csv                   # Generated results (after running)
```

## 🎓 Learning Outcomes

This project demonstrates:
- Data preprocessing and feature scaling
- Multiple classification algorithm implementation
- Model evaluation and comparison techniques
- Overfitting detection and prevention
- Data visualization best practices
- Cross-validation methodology
- Performance metrics interpretation

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a machine learning student project for heart disease prediction analysis.

## 🙏 Acknowledgments

- Dataset sourced from UCI Machine Learning Repository
- Built using Scikit-learn and modern ML best practices

---

**Note**: This is a student project for educational purposes. Models achieve realistic accuracy (~85-90%) and are not intended for actual medical diagnosis.
