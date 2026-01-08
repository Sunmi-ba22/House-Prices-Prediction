# House Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Key Findings](#key-findings)
- [Models & Results](#models--results)
- [Technical Implementation](#technical-implementation)
- [Business Insights](#business-insights)
- [Installation & Usage](#installation--usage)
- [Future Improvements](#future-improvements)
  

---

##  Overview

This project builds a **regression model** to predict house prices based on property features.This demonstrates my progression from classification to regression problems essential skills for quantitative finance roles.

**Problem Statement:** Can we accurately predict house prices using property characteristics?

**Business Value:** Enables real estate companies, banks, and investors to:
- Estimate property valuations automatically
- Identify undervalued/overvalued properties
- Make data-driven investment decisions
- Assess mortgage risk

---

## 📊 Dataset

**Source:** Housing.csv  
**Size:** 545 houses  
**Target Variable:** Price (continuous)

### Features (13 total)

**Numerical Features:**
- `area` - Property size in square feet
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `stories` - Number of floors
- `parking` - Parking spaces available

**Categorical Features:**
- `mainroad` - On main road (yes/no)
- `guestroom` - Has guest room (yes/no)
- `basement` - Has basement (yes/no)
- `hotwaterheating` - Has hot water heating (yes/no)
- `airconditioning` - Has AC (yes/no)
- `prefarea` - In preferred area (yes/no)
- `furnishingstatus` - Furnished/Semi-furnished/Unfurnished

### Data Quality
✅ No missing values  
✅ No duplicate records  
✅ Clean, well-structured dataset

---

## 🔍 Key Findings

### 1. Price Distribution
- **Mean Price:** $4,766,729
- **Median Price:** $4,340,000
- **Range:** $1.75M - $13.3M
- **Distribution:** Right-skewed with some high-value outliers

### 2. Feature Correlations with Price

| Feature | Correlation | Strength |
|---------|-------------|----------|
| Area | 0.54 | Strong |
| Bathrooms | 0.51 | Strong |
| Stories | 0.28 | Moderate |
| Parking | 0.27 | Moderate |
| Bedrooms | 0.26 | Moderate |

**Key Insight:** Property size (area) is the dominant price predictor.

### 3. Categorical Feature Impact

| Feature | Price Impact |
|---------|--------------|
| Air Conditioning | +29% average |
| Preferred Area | +24% average |
| Furnished Status | +15-20% premium |
| Basement | +12% average |

---

##  Models & Results

### Models Implemented

#### 1. Linear Regression (Baseline)
Simple model assuming linear relationships between features and price.

**Performance:**
- **R² Score:** 0.678 (67.8% variance explained)
- **MAE:** $847,326
- **RMSE:** $1,087,492

#### 2. Random Forest Regressor (Complex)
Ensemble of 100 decision trees capturing non-linear patterns.

**Performance:**
- **R² Score:** 0.614 (61.4% variance explained)
- **MAE:** $892,145
- **RMSE:** $1,143,287

### Best Model: Linear Regression

**Why Linear Regression Outperformed:**

1. **Dataset Size:** Only 545 houses—small for complex models
2. **Linear Relationships:** Price relationships are mostly linear
3. **Generalization:** Simpler model avoids overfitting
4. **Interpretability:** Clear feature coefficients for stakeholders

**Key Learning:** Complex models ≠ better performance. Start simple!

### Model Comparison

```
┌─────────────────────┬────────────┬────────────┬─────────┐
│ Metric              │ Linear Reg │ Random Forest│ Winner  │
├─────────────────────┼────────────┼──────────────┼─────────┤
│ R² Score            │   0.678    │    0.614     │   LR    │
│ MAE                 │  $847k     │   $892k      │   LR    │
│ RMSE                │  $1,087k   │   $1,143k    │   LR    │
│ Training Time       │   <1s      │    ~5s       │   LR    │
│ Interpretability    │   High     │    Low       │   LR    │
└─────────────────────┴────────────┴──────────────┴─────────┘
```

---

## Technical Implementation

### Technologies Used
```python
pandas==1.5.3         # Data manipulation
numpy==1.24.3         # Numerical computing
scikit-learn==1.2.2   # Machine learning
matplotlib==3.7.1     # Visualization
seaborn==0.12.2       # Statistical visualization
```

### Methodology

**1. Exploratory Data Analysis**
- Statistical summary and distributions
- Correlation analysis
- Outlier detection
- Feature relationship visualization

**2. Data Preprocessing**
```python
# Binary encoding: yes/no → 1/0
binary_features = ['mainroad', 'guestroom', 'basement', 
                   'hotwaterheating', 'airconditioning', 'prefarea']

# One-hot encoding: furnishing status
pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
```

**3. Train-Test Split**
- 80% training (436 houses)
- 20% testing (109 houses)
- Random seed: 42 (reproducibility)

**4. Model Training & Evaluation**
- Trained both Linear Regression and Random Forest
- Evaluated using R², MAE, RMSE
- Selected best model based on test performance

**5. Feature Importance Analysis**
```python
Top 5 Features (Random Forest):
1. Area          - 48.2%
2. Bathrooms     - 12.5%
3. Bedrooms      - 8.2%
4. Stories       - 6.1%
5. Parking       - 4.9%
```

---

## 💼 Business Insights

### For Real Estate Professionals

**Price Drivers:**
- Every 1,000 sq ft ≈ +$600-800k in value
- Additional bathroom > additional bedroom (12.5% vs 8.2% importance)
- AC and preferred location = major value multipliers

**Market Segmentation:**

| Segment | Price Range | Characteristics |
|---------|-------------|-----------------|
| Budget | <$3M | 2-3 bed, 2,500-3,500 sqft, unfurnished |
| Mid-Range | $3-6M | 3-4 bed, 4,000-6,000 sqft, semi-furnished |
| Premium | >$6M | 4+ bed, 6,500+ sqft, fully furnished, all amenities |

**Investment Strategy:**
- Focus on properties in preferred areas with AC (fastest appreciation)
- Bathroom count often undervalued by sellers
- Large area premium justified by 48% feature importance

---

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the analysis**
```bash
jupyter notebook house_price_analysis.ipynb
```

### Quick Prediction

```python
import pandas as pd
from joblib import load

# Load trained model
model = load('linear_regression_model.pkl')

# New house data
new_house = pd.DataFrame({
    'area': [5000],
    'bedrooms': [3],
    'bathrooms': [2],
    # ... other features
})

# Predict
predicted_price = model.predict(new_house)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

---

## 📈 Results Visualization
** See images folder




---

## Future Improvements

### Short-term Enhancements
- [ ] Feature engineering (area × bathrooms interaction)
- [ ] Hyperparameter tuning for Random Forest
- [ ] Cross-validation for robust performance estimates
- [ ] Outlier treatment strategies

### Long-term Additions
- [ ] Advanced models (XGBoost, Gradient Boosting)
- [ ] Additional features (location coordinates, property age)
- [ ] Time series component (historical prices)
- [ ] Web app deployment (Streamlit/Flask)
- [ ] A/B testing framework for model updates

---

## 🎯 Skills Demonstrated

### Technical Skills
✅ Regression analysis  
✅ Feature engineering  
✅ Model selection & comparison  
✅ Overfitting detection  
✅ Data preprocessing  
✅ Statistical analysis  
✅ Data visualization  

### Business Skills
✅ Real estate valuation  
✅ Feature importance interpretation  
✅ Stakeholder communication  
✅ Investment insights  


---

**Concepts Covered:**
- Linear vs non-linear regression
- Bias-variance tradeoff
- Feature correlation analysis
- Model interpretability
- Overfitting vs underfitting



---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### About This Project

This is **Project #2** in my data science learning journey toward a career in quantitative finance. 

---


## 🙏 Acknowledgments

- Dataset source: Kaggle Housing Prices Dataset
- Inspiration: Real estate valuation methods


---

## ⭐ Star This Repo

If you found this project helpful, please consider giving it a star! It helps others discover the project and motivates me to create more data science content
