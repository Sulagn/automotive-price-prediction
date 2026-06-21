> 📓 **Note:** GitHub's notebook renderer is currently experiencing a known, widespread issue affecting many repositories. If the notebook doesn't display below, view it here instead: [Open in nbviewer](https://nbviewer.org/github/Sulagn/automotive-price-prediction/blob/main/notebooks/automotive_price_prediction.ipynb)
# Automotive Price Prediction — VW & Audi

Predicting used car prices using Random Forest Regression, benchmarked against Linear Regression, on real-world VW and Audi listings.

## Overview

This project builds a machine learning pipeline to predict the resale price of used Volkswagen and Audi vehicles based on specifications like mileage, engine size, fuel type, transmission, and age. The goal was not just to train a model, but to understand *why* it performs the way it does — through exploratory data analysis, feature importance investigation, and a baseline model comparison.

## Dataset

- **Source:** VW and Audi used car listings (UK market)
- **Size:** 25,825 rows combined (10,668 Audi + 15,157 VW), reduced to 25,753 after cleaning
- **Features:** model, year, transmission, mileage, fuel type, tax, mpg, engine size, brand
- **Target:** price (£)

## Process

1. **Data cleaning** — merged two brand-specific datasets, added a `brand` column, identified and removed 72 rows with invalid `engineSize = 0` (data entry errors — confirmed these weren't electric vehicles by checking `fuelType`)
2. **Exploratory Data Analysis** — visualized price distribution (right-skewed), examined the non-linear relationship between mileage and price
3. **Feature engineering** — one-hot encoded categorical variables (model, transmission, fuelType, brand)
4. **Modeling** — trained a Random Forest Regressor (100 trees), benchmarked against Linear Regression as a baseline
5. **Evaluation** — R², MAE, feature importance analysis, and actual-vs-predicted visualizations for both models

## Results

| Model | R² Score | MAE |
|---|---|---|
| Linear Regression (baseline) | 0.8851 | £2,294.55 |
| **Random Forest** | **0.9580** | **£1,306.43** |

Random Forest reduced average prediction error by ~43% compared to the linear baseline.

## Key Findings

- **MPG was the strongest predictor of price (54% feature importance)** — not because fuel efficiency directly drives price, but because it acts as a proxy for vehicle class: high-performance/luxury cars have low mpg and high price, economy cars have high mpg and low price. Verified with a dedicated Price vs MPG scatter plot.
- **The relationship between mileage and price is non-linear** — prices drop sharply with initial mileage, then plateau. This explains why Random Forest meaningfully outperformed Linear Regression, which can only model straight-line relationships.
- **Linear Regression produced invalid negative price predictions** for some low-price cars — a structural limitation absent in Random Forest, which can only predict within the range of values seen during training.
- **Both models underpredict rare, high-end vehicles (£80k+)** — likely due to limited training examples in that price range, a known limitation worth addressing with more luxury-vehicle data in future work.

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (RandomForestRegressor, LinearRegression, train_test_split, metrics)
- Matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook notebooks/automotive_price_prediction.ipynb
```

## Future Improvements

- Address high-price underprediction with additional luxury vehicle data or a separate model for that segment
- Try gradient boosting (XGBoost/LightGBM) as a further benchmark
- Hyperparameter tuning via GridSearchCV
