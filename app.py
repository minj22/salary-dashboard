import streamlit as st
import pandas as pd
from model import SalaryPredictor, MULTI

st.set_page_config(page_title='Developer Salary Predictor',page_icon='📊',layout='wide')
st.title('Developer Salary Predictor')
st.caption('MIE1624 Assignment 2 · Ordinal logistic regression · Stack Overflow 2026')
st.info('Predictions reflect associations in survey responses, not causal effects or guaranteed salaries.')

@st.cache_resource(show_spinner='Training final ordinal model (first launch may take a few minutes)…')
def get_predictor():
    return SalaryPredictor()

predictor=get_predictor()
st.subheader('Your survey responses')
st.write('Answer the original survey questions below. Leave an item unanswered if it does not apply.')
answers={}
with st.form('survey'):
    left,right=st.columns(2)
    for i,question in enumerate(predictor.selected_questions):
        with (left if i%2==0 else right):
            if question in predictor.numeric_cols:
                raw=st.text_input(question,placeholder='Enter a number, or leave blank',key=question)
                try: answers[question]=float(raw) if raw.strip() else None
                except ValueError:
                    st.warning(f'{question}: please enter a valid number.')
                    answers[question]=None
            elif question in MULTI:
                chosen=st.multiselect(question,predictor.options[question],key=question)
                answers[question]=';'.join(chosen) if chosen else None
            else:
                chosen=st.selectbox(question,['(Not answered)']+predictor.options[question],key=question)
                answers[question]=None if chosen=='(Not answered)' else chosen
    submitted=st.form_submit_button('Predict salary and recommend skills',type='primary',use_container_width=True)

if submitted:
    bucket,probs=predictor.predict(answers)
    expected=sum(i*p for i,p in enumerate(probs))
    st.divider()
    st.subheader('Predicted salary distribution')
    a,b=st.columns(2)
    a.metric('Most likely bucket',f'{bucket} — {predictor.bucket_labels[bucket]}')
    b.metric('Expected bucket index',f'{expected:.4f}')
    chart=pd.DataFrame({'Salary bucket':range(len(probs)),'Probability':probs}).set_index('Salary bucket')
    st.bar_chart(chart)
    with st.expander('All bucket labels and probabilities'):
        st.dataframe(pd.DataFrame({'Bucket':range(len(probs)),
                                   'Salary range':predictor.bucket_labels,
                                   'Probability':probs}),hide_index=True)
    st.subheader('Ranked technology recommendations')
    recommendations=predictor.recommend(answers)
    st.dataframe(recommendations,hide_index=True,use_container_width=True)
    if len(recommendations):
        positive=recommendations[recommendations['Predicted change']>0]
        st.write(f'{len(positive)} of {len(recommendations)} unreported candidate technologies have a positive predicted change.')
        if len(positive)<3:
            st.warning('Fewer than three candidates have positive predicted improvement. Negative values are not beneficial recommendations.')
    st.caption('Change = difference in expected ordinal salary-bucket index when adding a technology while holding other responses fixed. Not a causal salary estimate.')
