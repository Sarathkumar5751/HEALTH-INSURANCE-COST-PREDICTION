# Health Insurance Cost Prediction (Go)

This project implements a **Random Forest Regressor from scratch in Go** to predict health insurance charges.  
The model is trained on the [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance), which includes features like age, sex, BMI, children, smoking status, and region.

---

## ✨ Features
- Preprocesses the dataset (`insurance.csv`)
- Encodes categorical variables (sex, smoker, region)
- Splits data into **train/test sets** (default 80/20)
- Implements **Random Forest Regression** in pure Go
- Evaluates performance using **RMSE**
- Saves the trained model to `rf_model.gob`

---

## ⚙️ Requirements
- Go 1.18+ (tested on Go 1.25.1)
- Dataset file: `insurance.csv` (Kaggle)

---

## 🚀 Run Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/health_insurance_project.git
   cd health_insurance_project

