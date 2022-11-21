import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

### Modelling 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import joblib

### Remove unnecessary warnings
import warnings
warnings.filterwarnings('ignore')


px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = 'reds'

st.header('Aplikasi Prediksi Man of the Match Fifa World Cup')

def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/andreandriand/dataset/main/FIFA%202018%20Statistics.csv')
    return df

data = load_data()


labels = data["Man of the Match"]
X = data.drop(columns=["Man of the Match", "Date", "Team", "Opponent", "Blocked", "Corners", "Free Kicks", "1st Goal", "Goals in PSO", "Own goal Time"])
X = X.replace(np.nan, 0)

st.subheader('Dataset Statistic Fifa World Cup 2018')

st.write(data)


# Sidebar

st.sidebar.header('User Input Features')

with st.form(key ='Form1'):
    with st.sidebar:
        goal = st.sidebar.number_input("Goal Scored", min_value=0)   
        possession = st.sidebar.number_input("Ball Possession (%)", min_value=0, max_value=100)   
        attempt = st.sidebar.number_input("Attempts", min_value=0)   
        onTarget = st.sidebar.number_input("Attempt On Target", min_value=0)   
        offTarget = st.sidebar.number_input("Attempt Off Target", min_value=0)   
        offsides = st.sidebar.number_input("Offsides", min_value=0)
        saves = st.sidebar.number_input("Saves", min_value=0)
        passAcc = st.sidebar.number_input("Pass Accuracy (%)", min_value=0, max_value=100)
        passMade = st.sidebar.number_input("Passes Made", min_value=0)
        distance = st.sidebar.number_input("Distance Covered (Kms)", min_value=0)
        fouls = st.sidebar.number_input("Fouls Committed", min_value=0)
        rounds = st.sidebar.selectbox('Round', ('Group Stage', 'Round of 16', 'Quarter Finals', 'Semi- Finals', '3rd Place', 'Final'))
        ownGoal = st.sidebar.number_input("Own Goal", min_value=0)
        submitted1 = st.form_submit_button(label = 'Predict')

if submitted1:

    st.subheader('Hasil Prediksi')
    
    modelGaussian = joblib.load('nb.joblib')
    modelKNN = joblib.load('knn.joblib')
    modelTree = joblib.load('tree.joblib')
    scaler = joblib.load('scaler.joblib')

    normalize = scaler.transform([[goal, ownGoal, attempt, onTarget, offTarget, possession, saves, fouls, offsides, passAcc, passMade, distance]]).tolist()[0]

    phase = [0,0,0,0,0,0]
    if rounds == 'Group Stage':
        phase[0] = 1
    elif rounds == 'Round of 16':
        phase[1] = 1
    elif rounds == 'Quarter Finals':
        phase[2] = 1
    elif rounds == 'Semi Finals':
        phase[3] = 1
    elif rounds == 'Final':
        phase[4] = 1
    else:
        phase[5] = 1


    inputs = {
        'goal': normalize[0],
        'ownGoal': normalize[1],
        'attempt': normalize[2],
        'onTarget': normalize[3],
        'offTarget': normalize[4],
        'possession': normalize[5],
        'saves': normalize[6],
        'fouls': normalize[7],
        'offsides': normalize[8],
        'passAcc': normalize[9],
        'passMade': normalize[10],
        'distance': normalize[11],
        'Group Stage': phase[0],
        'Round of 16': phase[1],
        'Quarter Finals': phase[2],
        'Semi Finals': phase[3],
        'Final': phase[4],
        'Third Place': phase[5]
    }

    inputs = np.array([[val for val in inputs.values()]])

    tree = joblib.load("tree.joblib")
    pred = tree.predict(inputs)
    if pred[0] == "Yes":
        st.write("Tree: Man of the Match")
    else:
        st.write("Tree: Not Man of the Match")

    knn = joblib.load("knn.joblib")
    pred1 = knn.predict(inputs)
    if pred1[0] == "Yes":
        st.write("KNN: Man of the Match")
    else:
        st.write("KNN: Not Man of the Match")

    gaussian = joblib.load("nb.joblib")
    pred2 = gaussian.predict(inputs)
    if pred2[0] == "Yes":
        st.write("Naive Bayes: Man of the Match")
    else:
        st.write("Naive Bayes: Not Man of the Match")