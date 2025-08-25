<p align="center">
  <img src="assets/title_image.png" width="65%" style="border-radius: 10px;">
</p>

<h1 align="center">📊 Rossmann Store Sales Prediction</h1>

<p align="center" style="font-size: 16px;">


# 🚀 Rossmann Sales Forecasting — End-to-End ML with AWS SageMaker & CI/CD

An end-to-end **Machine Learning + MLOps** project to forecast daily sales for thousands of drug stores.  
This project highlights **CI/CD automation** and **AWS cloud services** (S3, SageMaker, API Gateway, IAM) for model training, deployment, and inference at scale.

---

## 📌 Problem Statement  
Rossmann, one of Europe’s largest drugstore chains, aims to accurately forecast daily sales for each of its stores.  
The business need is to optimize **inventory**, **workforce**, and **promotional strategies** through data-driven sales forecasting.  

---

## 🎯 Project Aim  
Build a **scalable, reproducible, and automated ML pipeline** that:  
- Preprocesses raw sales & store data  
- Trains and tunes ML models on AWS SageMaker  
- Deploys endpoints with CI/CD pipelines  
- Serves **real-time predictions via API Gateway** and supports **batch transform jobs**  

---

## 🔍 Project Objectives  

### 🛠️ Data & Features  
- 📦 Preprocess and merge training & store data (SageMaker Processing jobs)  
- 🔧 Feature engineering: time-based, promotions, competition, holidays  
- 📊 EDA on trends, seasonality, promotions, holidays  

### 🤖 Modeling  
- Train multiple regressors (Random Forest, XGBoost, Gradient Boosting, Linear models)  
- Evaluate with RMSE, MAE, R²  
- Select top features with Random Forest & XGBoost importances  
- Hyperparameter tuning with **SageMaker HPO jobs**  

### 🚀 Deployment & Inference  
- Train & deploy best model using **SageMaker Training + Hosting**  
- Serve inference through a **SageMaker Endpoint**  
- Enable **real-time API predictions via API Gateway**  
- Support **batch transform jobs** for bulk CSV predictions  

---

## 🧾 Dataset Overview  
The project uses two datasets (`train.csv`, `store.csv`) with the following key columns:  

| Column | Description |
|--------|-------------|
| Store  | Unique store ID |
| Date   | Daily record date |
| Sales  | Target variable |
| Promo  | Promotion flag |
| StateHoliday | Holiday flag |
| SchoolHoliday | School closure flag |
| StoreType | Store classification |
| Assortment | Assortment type |
| CompetitionDistance | Distance to nearest competitor |
| Promo2 | Ongoing promotion campaign |

Engineered features:  
- Year, Month, Day, WeekOfYear  
- IsWeekend, IsPromoMonth, Promo2Active  
- CompetitionOpenTimeMonths  

---

## 📊 Model Performance  

We benchmarked multiple algorithms on both **all features** and **selected features** (post-feature selection).  

| Model | RMSE (all) | MAE (all) | R² (all) | RMSE (selected) | MAE (selected) | R² (selected) |
|-------|------------|-----------|----------|-----------------|----------------|---------------|
| Linear Regression | 2807.31 | 2046.23 | 0.1837 | 2826.61 | 2057.58 | 0.1725 |
| Ridge | 2807.31 | 2046.23 | 0.1837 | 2826.61 | 2057.58 | 0.1725 |
| Lasso | 2807.31 | 2046.21 | 0.1837 | 2826.61 | 2057.57 | 0.1725 |
| Random Forest | **989.67** | **616.37** | **0.8986** | 1043.66 | 659.08 | 0.8872 |
| Gradient Boosting | 2464.13 | 1801.99 | 0.3711 | 2456.93 | 1801.13 | 0.3748 |
| AdaBoost | 5696.37 | 5086.43 | -2.3608 | 5295.81 | 4599.30 | -1.9048 |
| XGBoost | 1213.96 | 873.29 | 0.8474 | 1223.66 | 878.51 | 0.8449 |

### 🏆 Winner: Random Forest  
- With **RMSE ≈ 989** and **R² ≈ 0.90**, Random Forest outperformed all other models.  
- XGBoost followed closely, but Random Forest provided **more stable results** across feature sets.  

---

## 🎛️ Hyperparameter Tuning (HPO) Story  

During SageMaker Hyperparameter Optimization (HPO):  
- Training on the **full dataset** caused **very large model artifacts** (hundreds of MBs)  
- Endpoints **failed due to memory allocation errors** when loading these oversized models  

To balance **performance vs scalability**, we made a **deliberate trade-off**:  
- Used **50% of the dataset** for HPO  
- This reduced training cost, model size, and ensured deployment stability  
- The best tuned model achieved:  


Validation RMSE ≈ 1091.83



---

## ⚙️ CI/CD Pipeline  

CI/CD automation was set up using **GitHub Actions** to handle:  
- **Preprocessing pipeline** → Trigger SageMaker processing jobs  
- **Training pipeline** → Trigger SageMaker training jobs  
- **HPO pipeline** → Hyperparameter tuning jobs  
- **Deployment pipeline** → Deploy trained model to a SageMaker endpoint  

Each pipeline is triggered on GitHub pushes to `main` branch with YAML workflows (`.github/workflows/`).  

---

## ☁️ AWS Infrastructure  

- **S3** → Storage for raw, processed, and model artifacts  
- **SageMaker** → Training, HPO, model hosting, and batch transform  
- **IAM** → Roles for SageMaker, API Gateway, and CI/CD permissions  
- **API Gateway** → Public-facing REST API for predictions  
- **CloudWatch** → Endpoint logs & monitoring  

---

## 📡 Inference Modes  

- **Real-Time Inference (API Gateway → SageMaker Endpoint)**  
  - Input: single row of features as CSV  
  - Output: predicted daily sales  

- **Batch Transform Jobs**  
  - Input: preprocessed test CSV in S3  
  - Output: predictions stored in S3  

---

## 🛠️ Tools & Technologies  

- **Languages/Libraries** → Python, Pandas, NumPy, Scikit-Learn, XGBoost, Joblib  
- **AWS Services** → S3, SageMaker, IAM, API Gateway, CloudWatch  
- **CI/CD** → GitHub Actions  
- **Infrastructure as Code** → Terraformer (import infra as `.tf`)  

---

## ✅ Key Takeaways  

- CI/CD pipelines automate preprocessing, training, tuning, and deployment.  
- AWS SageMaker scales training & inference seamlessly.  
- Real-time + batch predictions both supported.  
- API Gateway provides external access to deployed ML endpoints.  

---

## 🚀 Future Work  

- Add monitoring for **model drift** with CloudWatch metrics  
- Build a **model registry** for versioning  
- Automate retraining pipelines on new data  
- Extend Terraform configs for **SageMaker & API Gateway** resources  

---

## 🙋‍♂️ Author

**Abdullah Shahzad**  
📧 [abdullahhunjra@gmail.com](mailto:abdullahshahzadhunjra@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/abdullahhunjra)  
💻 [GitHub](https://github.com/abdullahhunjra)

---