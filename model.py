"""Deployment inference using the exact 50 selected Q2 features and Q4 ordinal model.
The training split, numeric preprocessing, and threshold estimator follow the notebook.
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
FEATURES = json.loads((HERE / 'features.json').read_text())
MULTI = ['RoleFunctions', 'LanguagesHaveWorkedWith', 'CloudPlatHaveWorkedWith',
         'KnowMgmtHaveWorkedWith', 'EmploymentAddl', 'PurchasePower',
         'LanguagesWantToWorkWith', 'AICodeAgentHaveWorkedWith', 'LearnCodeHow']
NUMERIC = ['YearsCode', 'JobSat']
THRESHOLDS = {'YearsCode':80, 'WorkExp':80, 'ToolCount':100, 'AIToolCount':100}
SKILLS = {'Bazel': ('CloudPlatHaveWorkedWith','Bazel'),
          'PHP': ('LanguagesHaveWorkedWith','PHP'),
          'Microsoft SharePoint': ('KnowMgmtHaveWorkedWith','Microsoft SharePoint'),
          'Google Antigravity': ('AICodeAgentHaveWorkedWith','Google Antigravity')}

def bucket_lower_bound(bucket):
    match = re.findall(r'\$([\d,]+)', str(bucket))
    if not match: raise ValueError(f'Unknown salary bucket: {bucket}')
    return int(match[0].replace(',',''))

class OrdinalModel:
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self.models_ = []
        for threshold in self.classes_[:-1]:
            m = LogisticRegression(C=10.0, penalty='l2', solver='liblinear',
                                   max_iter=2000, random_state=42)
            m.fit(X, (np.asarray(y)>threshold).astype(int))
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        gt = np.column_stack([m.predict_proba(X)[:,1] for m in self.models_])
        gt = np.clip(np.minimum.accumulate(gt,axis=1),0,1)
        p = np.column_stack([1-gt[:,0], gt[:,:-1]-gt[:,1:], gt[:,-1]])
        p = np.clip(p,0,1)
        return p / p.sum(axis=1,keepdims=True)

class SalaryPredictor:
    def __init__(self, data_path=HERE/'processed_data_assignment2_2026.csv'):
        df = pd.read_csv(data_path,low_memory=False)
        labels = sorted(df['ConvertedCompYearlyBucket'].dropna().unique(),key=bucket_lower_bound)
        self.bucket_labels = labels
        df['target_encoded'] = df['ConvertedCompYearlyBucket'].map({v:i for i,v in enumerate(labels)})
        df = df[df['Employment'].isin(['Employed','Independent contractor, freelancer, or self-employed'])]
        excluded = ['ResponseId','Compensation','ConvertedCompYearly',
                    'ConvertedCompYearlyBucket','target_encoded']
        X = df.drop(columns=excluded)
        y = df['target_encoded']
        X_train,_,y_train,_ = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
        X_train = X_train.drop(columns=X_train.columns[X_train.isna().all()])
        self.numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
        self.raw_columns = X_train.columns.tolist()
        self.selected_questions = []
        for col in self.raw_columns:
            if col in FEATURES or any(f.startswith(col+'_') for f in FEATURES):
                self.selected_questions.append(col)
        # Keep the original training options, not transformed feature names.
        self.options = {}
        for q in self.selected_questions:
            if q in self.numeric_cols: continue
            values = X_train[q].dropna().astype(str)
            if q in MULTI:
                self.options[q] = sorted({v.strip() for row in values for v in row.split(';') if v.strip()})
            else:
                self.options[q] = sorted(set(values)|{'Missing'})
        # The notebook fits numeric imputation and scaling on all original numeric columns.
        numeric = X_train[self.numeric_cols].copy()
        for c, upper in THRESHOLDS.items():
            if c in numeric: numeric[c] = numeric[c].mask(numeric[c]>upper)
        self.imputer = SimpleImputer(strategy='median').fit(numeric)
        self.scaler = StandardScaler().fit(self.imputer.transform(numeric))
        self.model = OrdinalModel().fit(self.transform(X_train),y_train)

    def transform(self, rows):
        if isinstance(rows,dict): rows=pd.DataFrame([rows])
        rows=rows.reindex(columns=self.raw_columns)
        n=len(rows)
        numeric=rows[self.numeric_cols].apply(pd.to_numeric,errors='coerce')
        for c, upper in THRESHOLDS.items():
            if c in numeric: numeric[c]=numeric[c].mask(numeric[c]>upper)
        numeric_values=self.scaler.transform(self.imputer.transform(numeric))
        numeric_df=pd.DataFrame(numeric_values,columns=self.numeric_cols,index=rows.index)
        result=pd.DataFrame(index=rows.index)
        for feature in FEATURES:
            if feature in self.numeric_cols:
                result[feature]=numeric_df[feature].values
                continue
            if '__' in feature:
                q,option=feature.split('__',1)
                if option=='Missing':
                    result[feature]=rows[q].isna().astype(int).values
                else:
                    result[feature]=rows[q].fillna('').astype(str).map(
                        lambda s: int(option in [x.strip() for x in s.split(';')])
                    ).values
            else:
                # Single-choice one-hot feature: resolve using the original column names.
                matching=[q for q in self.raw_columns if feature.startswith(q+'_')]
                q=max(matching,key=len)
                category=feature[len(q)+1:]
                result[feature]=rows[q].fillna('Missing').astype(str).eq(category).astype(int).values
        return result[FEATURES]

    def predict(self, answers):
        probs=self.model.predict_proba(self.transform(answers))[0]
        bucket=int(np.argmax(probs))
        return bucket,probs

    def recommend(self, answers):
        _,baseline_probs=self.predict(answers)
        baseline=float(np.dot(np.arange(len(baseline_probs)),baseline_probs))
        rows=[]
        for name,(question,value) in SKILLS.items():
            existing=[x.strip() for x in str(answers.get(question) or '').split(';') if x.strip()]
            if value in existing: continue
            changed=dict(answers)
            changed[question]=';'.join(existing+[value])
            _,probs=self.predict(changed)
            expected=float(np.dot(np.arange(len(probs)),probs))
            rows.append({'Technology':name,'Expected bucket index':round(expected,4),
                         'Predicted change':round(expected-baseline,4)})
        return pd.DataFrame(rows).sort_values('Predicted change',ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=['Technology','Expected bucket index','Predicted change'])
