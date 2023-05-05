#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title('Predicting your placement')
model_random_forest=joblib.load('Forest_model.pkl')
model_logistic=joblib.load('Logr_model.pkl')
model_knn=joblib.load('knn_model.pkl')
model_decision_tree=joblib.load('dt_model.pkl')
model_svm=joblib.load('svm_model.pkl')
model_perceptron=joblib.load('perc_model.pkl')
model_naive_bayes=joblib.load('gnb_model.pkl')
model_back_prop=joblib.load('backprop_model.pkl')

model = st.selectbox('Select which model would like to test our prediction',('Logistic Regression', 'K Nearest Neighbours','Decision Trees', 'Gaussian Naive Bayes', 'Support Vector Machines', 'Random Forest', 'Perceptron Model', 'Back Propagation(MLP)'))

if model=='Logistic Regression':
    model=model_logistic
elif model=='K Nearest Neighbours':
    model=model_knn
elif model=='Decision Trees':
    model=model_decision_tree
elif model=='Gaussian Naive Bayes':
    model=model_naive_bayes
elif model=='Support Vector Machines':
    model=model_svm
elif model=='Random Forest':
    model=model_random_forest
elif model=='Perceptron Model':
    model=model_perceptron
elif model=='Back Propagation(MLP)':
    model=model_back_prop
    
age = st.text_input("Enter your age : ",0)
age=int(age)

gender = st.selectbox('What is your gender',('Male','Female'))
gender=1.0 if gender=='Male' else 0.0

stream = st.selectbox('What is your stream',('Electronics And Communication', 'Computer Science','Information Technology', 'Mechanical', 'Electrical', 'Civil'))
if stream=='Electronics And Communication':
    stream=3.0
elif stream=='Computer Science':
    stream=1.0
elif stream=='Information Technology':
    stream=4.0
elif stream=='Mechanical':
    stream=5.0
elif stream=='Electrical':
    stream=2.0
elif stream=='Civil':
    stream=0.0
    
internships = st.selectbox('How many internships have you done',('0','1','2','3'))
internships=int(internships)

CGPA=st.selectbox('What is your CGPA(round off to the nearest)',('5','6','7','8','9','10'))
CGPA=int(CGPA)

hostel = st.selectbox('Are you a hosteler',('Yes','No'))
hostel=1 if hostel=='Yes' else 0

backlog = st.selectbox('Did you ever had a backlog',('Yes','No'))
backlog=1 if backlog=='Yes' else 0

test=[age,gender,stream,internships,CGPA,hostel,backlog]
test =[test]
pred=model_random_forest.predict(test)

# st.markdown("""
# <style>
# .big-font {
#     font-size:50px !important;
# }
# </style>
# """, unsafe_allow_html=True)


if st.button('Predict'):
    if pred==0:
        st.markdown(f'<p style="background-color:#FF0000;color:#FFFFFF;font-size:24px;border-radius:2%;">You will not be placed</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p style="background-color:#FFFFFF;color:#33ff33;font-size:24px;border-radius:2%;">You will be placed</p>', unsafe_allow_html=True)                                                                                                   