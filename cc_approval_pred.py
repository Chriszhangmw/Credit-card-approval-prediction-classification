import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
#from secret import access_key, secret_access_key
import joblib
import streamlit as st
import boto3
import tempfile
import json
import requests
from streamlit_lottie import st_lottie_spinner
from train_model import  full_pipeline,process_train_model


train_original = pd.read_csv('https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/main/datasets/train.csv')

test_original = pd.read_csv('https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/main/datasets/test.csv')

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)
# full_data = full_data[:100]

def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()



def value_cnt_norm_cal(df,feature):
    '''
    Function to calculate the count of each value in a feature and normalize it
    '''
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat





model_type = st.selectbox("选择智能分析模型", ("sgd", "logistic_regression","decision_tree","random_forest","bagging","adaboost","extra_trees"))
if model_type:
    class_report,roc_curve_image,confusion_matrix_image,fearure_image = process_train_model(model_type)

    st.header("混淆矩阵:")
    st.image(confusion_matrix_image)

    st.header("测试报告:")
    st.image(class_report)

    st.header("ROC曲线")
    st.image(roc_curve_image)

    st.header("特征重要性")
    st.image(fearure_image)
    # st.bar_chart(fearure_image)


############################# Streamlit ############################

st.write("""
# 金融服务智能分析系统
该系统根据用户填报的信息，采用AI算法对用户建模与分析，预测出金融服务被审批通过与否的可能性，当用户填报完毕，点击“开始预测”即可
""")

#Gender input
st.write("""
## 性别
""")
input_gender = st.radio('选择你的性别',['Male','Female'], index=0)


# Age input slider
st.write("""
## 年龄
""")
input_age = np.negative(st.slider('选择你的年龄', value=42, min_value=18, max_value=70, step=1) * 365.25)




# Marital status input dropdown
st.write("""
## 婚姻状态
""")
marital_status_values = list(value_cnt_norm_cal(full_data,'Marital status').index)
marital_status_key = ['Married', 'Single/not married', 'Civil marriage', 'Separated', 'Widowed']
marital_status_dict = dict(zip(marital_status_key,marital_status_values))
input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)
input_marital_status_val = marital_status_dict.get(input_marital_status_key)


# Family member count
st.write("""
## 家庭人口数
""")
fam_member_count = float(st.selectbox('Select your family member count', [1,2,3,4,5,6]))


# Dwelling type dropdown
st.write("""
## 住宅信息
""")
dwelling_type_values = list(value_cnt_norm_cal(full_data,'Dwelling').index)
dwelling_type_key = ['House / apartment', 'Live with parents', 'Municipal apartment ', 'Rented apartment', 'Office apartment', 'Co-op apartment']
dwelling_type_dict = dict(zip(dwelling_type_key,dwelling_type_values))
input_dwelling_type_key = st.selectbox('Select the type of dwelling you reside in', dwelling_type_key)
input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)


# Income
st.write("""
## 收入
""")
input_income = np.int(st.text_input('Enter your income (in USD)',0))


# Employment status dropdown
st.write("""
## 就业状况
""")
employment_status_values = list(value_cnt_norm_cal(full_data,'Employment status').index)
employment_status_key = ['Working','Commercial associate','Pensioner','State servant','Student']
employment_status_dict = dict(zip(employment_status_key,employment_status_values))
input_employment_status_key = st.selectbox('Select your employment status', employment_status_key)
input_employment_status_val = employment_status_dict.get(input_employment_status_key)


# Employment length input slider
st.write("""
## 工作年限
""")
input_employment_length = np.negative(st.slider('Select your employment length', value=6, min_value=0, max_value=30, step=1) * 365.25)


# Education level dropdown
st.write("""
## 学历
""")
edu_level_values = list(value_cnt_norm_cal(full_data,'Education level').index)
edu_level_key = ['Secondary school','Higher education','Incomplete higher','Lower secondary','Academic degree']
edu_level_dict = dict(zip(edu_level_key,edu_level_values))
input_edu_level_key = st.selectbox('Select your education status', edu_level_key)
input_edu_level_val = edu_level_dict.get(input_edu_level_key)


# Car ownship input
st.write("""
## 是否用机动车
""")
input_car_ownship = st.radio('Do you own a car?',['Yes','No'], index=0)

# Property ownship input
st.write("""
## Property ownship
""")
input_prop_ownship = st.radio('Do you own a property?',['Yes','No'], index=0)


# Work phone input
st.write("""
## Work phone
""")
input_work_phone = st.radio('Do you have a work phone?',['Yes','No'], index=0)
work_phone_dict = {'Yes':1,'No':0}
work_phone_val = work_phone_dict.get(input_work_phone)

# Phone input
st.write("""
## Phone
""")
input_phone = st.radio('Do you have a phone?',['Yes','No'], index=0)
work_dict = {'Yes':1,'No':0}
phone_val = work_dict.get(input_phone)

# Email input
st.write("""
## Email
""")
input_email = st.radio('Do you have an email?',['Yes','No'], index=0)
email_dict = {'Yes':1,'No':0}
email_val = email_dict.get(input_email)

st.markdown('##')
st.markdown('##')
# Button
predict_bt = st.button('Predict')

# list of all the input variables
profile_to_predict = [0, # ID
                    input_gender[:1], # gender
                    input_car_ownship[:1], # car ownership
                    input_prop_ownship[:1], # property ownership
                    0, # Children count (which will be dropped in the pipeline)
                    input_income, # Income
                    input_employment_status_val, # Employment status
                    input_edu_level_val, # Education level
                    input_marital_status_val, # Marital status
                    input_dwelling_type_val, # Dwelling type
                    input_age, # Age
                    input_employment_length,    # Employment length
                    1, # Has a mobile phone (which will be dropped in the pipeline)
                    work_phone_val, # Work phone
                    phone_val, # Phone
                    email_val,  # Email
                    'to_be_droped', # Job title (which will be dropped in the pipeline)
                    fam_member_count,  # Family member count
                    0.00, # Account age (which will be dropped in the pipeline)
                    0 # target set to 0 as a placeholder
                    ]



profile_to_predict_df = pd.DataFrame([profile_to_predict],columns=train_copy.columns)




# add the profile to predict as a last row in the train data
train_copy_with_profile_to_pred = pd.concat([train_copy,profile_to_predict_df],ignore_index=True)

# whole dataset prepared
train_copy_with_profile_to_pred_prep = full_pipeline(train_copy_with_profile_to_pred)

# Get the row with the ID = 0, and drop the ID, and target(placeholder) column
profile_to_pred_prep = train_copy_with_profile_to_pred_prep[train_copy_with_profile_to_pred_prep['ID'] == 0].drop(columns=['ID','Is high risk'])




#Animation function
@st.experimental_memo
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_loading_an = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json')


def make_prediction(model_name):
    model_file_path = 'saved_models/{0}/{0}_model.sav'.format(model_name)
    model =joblib.load(model_file_path)
    # prediction from the model on AWS S3
    return model.predict(profile_to_pred_prep)

if predict_bt:
    with st_lottie_spinner(lottie_loading_an, quality='high', height='200px', width='200px'):
        final_pred = make_prediction()
    # if final_pred exists, then stop displaying the loading animation
    if final_pred[0] == 0:
        st.success('## You have been approved for a credit card')
        st.balloons()
    elif final_pred[0] == 1:
        st.error('## Unfortunately, you have not been approved for a credit card')








