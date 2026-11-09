# Developer Salary Predictor — Streamlit

This app reconstructs the preprocessing and final 50-feature ordinal model from `assignment2 (3).ipynb` using the original Stack Overflow 2026 assignment CSV. It uses the same 80/20 stratified split (`random_state=42`) and final Q4 settings (`C=10`, `penalty=l2`, `liblinear`), without running grid search.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```
