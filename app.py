#### LOCAL APP
#### Linkedin application
#### Import packages
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#### Add header to describe app
st.markdown("# Are you a Linkedin User?")
st.markdown("By Patrick Huang")
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x

ss = pd.DataFrame({
    "sm_li": (clean_sm(s["web1h"])),
    "income": np.where(s["income"] <= 9,s["income"], np.nan),
    "education": np.where(s["educ2"] <= 8,s["educ2"], np.nan),
    "parent": np.where(s["par"] == 1, 1, 0),
    "married": np.where(s["marital"] == 1, 1, 0),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] <= 98,s["age"], np.nan)})

ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married" , "female", "age" ]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987)

lr = LogisticRegression(class_weight='balanced')

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")



pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


#Inputs
income = st.selectbox("Income level", 
             options = ["Less than $10,000",
                        "10 to under $20,000",
                        "20 to under $30,000",
                        "30 to under $40,000",
                        "40 to under $50,000",
                        "50 to under $75,000",
                        "75 to under $100,000",
                        "100 to under $150,000",
                        "$150,000 or more",
                        ])

if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
    income = 5
elif income == "50 to under $75,000":
    income = 6
elif income == "75 to under $100,000":
    income = 7
elif income == "100 to under $150,000":
    income = 8
elif income == "$150,000 or more":
    income = 9
else: 
    income = 99

education = st.selectbox("Education level", 
             options = ["Less than high school (Grades 1-8 or no formal schooling)",
                        "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                        "High school graduate (Grade 12 with diploma or GED certificate)",
                        "Some college, no degree (includes some community college)",
                        "Two-year associate degree from a college or university",
                        "Four-year college or university degree (e.g., BS, BA, AB)",
                        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
                        "Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)",
                        ])

if education == "Less than high school (Grades 1-8 or no formal schooling)":
    education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education = 3
elif education == "Some college, no degree (includes some community college)":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education == "Four-year college or university degree (e.g., BS, BA, AB)":
    education = 6
elif education == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    education = 7
elif education == "Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)":
    education = 8
else: 
    education = 99
    
parent = st.selectbox("Are you a Parent", 
             options = ["Yes",
                        "No",
                        ])

if parent == "Yes":
    parent = 1
elif parent == "No":
    parent = 2
else: 
    parent = 9
    
married = st.selectbox("Marital Status", 
             options = ["Married",
                        "Living with a partner",
                        "Divorced",
                        "Separated",
                        "Widowed",
                        "Never been married",
                        ])

if married == "Married":
    married = 1
elif married == "Living with a partner":
    married = 2
elif married == "Divorced":
    married = 3
elif married == "Separated":
    married = 4
elif married == "Widowed":
    married = 5
elif married == "Never been married":
    married = 6
else: 
    married = 99


female = st.selectbox("Whats your Gender?", 
             options = ["Male",
                        "Female",
                        "Other",
                        ])

if female == "Male":
    female = 1
elif female == "Female":
    female = 2
elif female == "Other":
    female = 3
else: 
    female = 9

age = st.slider(label="What's your Age", 
          min_value=1,
          max_value=97,
          value=0)

person = [income, education, parent , married , female, age]
# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])


st.markdown(f"Are you a User?: {predicted_class[0]}") # 0=not user, 1=user
st.markdown(f"Probability that this person is linkedin user: {probs[0][1]}")
st.markdown("0 = not user, 1 = user")




