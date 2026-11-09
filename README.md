# Developer Salary Predictor — Streamlit

This app reconstructs the preprocessing and final 50-feature ordinal model from `assignment2 (3).ipynb` using the original Stack Overflow 2026 assignment CSV. It uses the same 80/20 stratified split (`random_state=42`) and final Q4 settings (`C=10`, `penalty=l2`, `liblinear`), without running grid search.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Upload all files in this folder to a **private** GitHub repository (the CSV is included).
2. Visit https://share.streamlit.io and create an app from that repository, selecting `app.py`.
3. Wait for the first launch to fit the final model. Subsequent reruns use `st.cache_resource`.
4. Test input changes, probability sums, and technology rankings.

**Dataset caution:** The assignment's CSV is approximately 71 MB and may be subject to course/data sharing restrictions. Before using public hosting, confirm that uploading the dataset to GitHub or a third-party service is permitted. A private repository alone does not make the hosted data private from the hosting provider. Do not publish the CSV publicly without authorization.

**Important:** The assignment notebook itself should remain unchanged; its restrictions on saving and reloading intermediate data are separate from this deployment application. The deployment app retrains the fixed final model from the original CSV at startup, without serialized artifacts.

**Limitations:** Four technology candidates are inherited from Q5.5; some may have negative predicted changes. The dashboard labels these explicitly. The original survey wording is not provided in the processed CSV, so labels use column names rather than invented question text. Validate the deployed output against the notebook's Q5.4/Q5.5 before submitting.
