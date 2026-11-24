# Medical Insurance Cost & Risk Prediction

End-to-end machine learning project on a 100,000-row medical insurance dataset (54 features): exploratory analysis, two predictive models, and a Streamlit app that serves them.

- **Regression:** predicts a person's annual medical cost.
- **Classification:** predicts whether a person falls into the high-cost (high-risk) group, defined as annual cost above the median.

## Results

| Task | Model | Metric |
|---|---|---|
| Regression | Linear Regression | R² 0.44 · MAE $1,345 |
| Regression | Random Forest | R² 0.65 · MAE $985 |
| Regression | **Gradient Boosting** (selected) | **R² 0.66 · MAE $960** |
| Classification | Random Forest | Accuracy 0.79 · F1 0.79 |
| Classification | **LightGBM** (selected) | **Accuracy 0.81 · F1 0.81** |

Columns that directly reveal the target (`annual_premium`, `monthly_premium`, `total_claims_paid`) are dropped before training to avoid leakage.

## Workflow

1. **Exploration:** distributions, correlations, outlier visualization.
2. **Cleaning:** missing value imputation, removal of invalid rows.
3. **Feature engineering:** one-hot encoding, scaling.
4. **Modeling:** baseline vs. tree-based models, compared on a held-out 20% test split.
5. **Serving:** trained models saved with `joblib` and loaded by the Streamlit app.

## Project structure

```
insurance_model.ipynb     # EDA, preprocessing, training, evaluation
final_app.py              # Streamlit app (cost + risk prediction)
medical_insurance.csv     # dataset
regresyon_model.pkl       # Gradient Boosting regressor   (Git LFS)
siniflandirma_model.pkl   # LightGBM classifier            (Git LFS)
```

## Run locally

```bash
git lfs install && git clone https://github.com/Elifglm/medical_insurance_project.git
cd medical_insurance_project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run final_app.py
```

## Tech stack

Python · pandas · scikit-learn · LightGBM · matplotlib / seaborn · Streamlit
