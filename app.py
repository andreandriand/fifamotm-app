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

tab1, tab2, tab3 = st.tabs(["Dataset", "Preprocessing", "Implementasi"])

def load_data():
        df = pd.read_csv('https://raw.githubusercontent.com/andreandriand/dataset/main/FIFA%202018%20Statistics.csv')
        return df

data = load_data()

data['Man of the Match'] = data['Man of the Match'].replace({'Yes' : 1, 'No' : 0})

with tab1:
    st.header('Aplikasi Prediksi Man of the Match Fifa World Cup')

    st.write("Aplikasi ini dibuat untuk memprediksi pemain yang akan menjadi Man of the Match atau pemain terbaik dalam suatu pertandingan piala dunia. Aplikasi ini memanfaatkan dataset yang berisi data statistik dari 128 pertandingan piala dunia 2018. Dataset ini dapat diakses pada link berikut: https://www.kaggle.com/mathan/fifa-2018-match-statistics")
    st.write("Aplikasi ini dibuat menggunakan bahasa pemrograman Python dan menggunakan beberapa library seperti Streamlit, Pandas, Numpy, dan Plotly. Serta menggunakan algoritma decision Tree, Naive Bayes, dan KNN. \n Untuk menggunakan aplikasi ini, anda harus memberikan inputan berupa Goal, Ball Possession, Attempt, Shot on Target, Shot off Target, Offside, Saves, Pass Accuracy, Passes Made, Distance Covered, Fouls Commited, Round, dan Own Goal \n Setelah memasukkan inputan, silahkan klik tombol 'Predict' untuk mendapatkan hasil prediksi.")

    st.subheader('Dataset Statistic Fifa World Cup 2018')

    st.write(data)

with tab2:
    st.subheader('Preprocessing Data')
    st.write('Hilangkan fitur yang tidak diperlukan:')
    st.write('Man of the Match, Date, Team, Opponent, Blocked, Corners, Free Kicks, 1st Goal, PSO, Goals in PSO, Own goal Time, Yellow Card, Yellow & Red, Red')
    labels = data["Man of the Match"]
    X = data.drop(columns=["Man of the Match", "Date", "Team", "Opponent", "Blocked", "Corners", "Free Kicks", "1st Goal", "PSO", "Goals in PSO", "Own goal Time", "Yellow Card", "Yellow & Red", "Red"])
    X = X.replace(np.nan, 0)
    st.write(X)

    split_overdue_round = pd.get_dummies(X["Round"], prefix="Round")
    X = X.join(split_overdue_round)

    X = X.drop(columns = "Round")

    old_normalize_feature_labels = ['Goal Scored', 'Ball Possession %', 'Attempts', 'On-Target', 'Off-Target', 'Offsides', 'Saves', 'Pass Accuracy %', 'Passes', 'Distance Covered (Kms)', 'Fouls Committed', 'Own goals']
    new_normalized_feature_labels = ['Norm_Goal Scored', 'Norm_Ball Possession %', 'Norm_Attemps', 'Norm_On-Target', 'Norm_Off-Target', 'Offsides', 'Saves', 'Pass Accuracy', 'Passes', 'Distance Covered', 'Fouls Committed', 'Own Goals']
    normalize_features = data[old_normalize_feature_labels].replace(np.nan, 0)

    scaler = joblib.load('scaler.joblib')
    scaler.fit(normalize_features)
    MinMaxScaler()
    normalized_features = scaler.transform(normalize_features)
    normalized_feature_data = pd.DataFrame(normalized_features, columns = new_normalized_feature_labels)

    X = X.drop(columns = old_normalize_feature_labels)
    X = X.join(normalized_feature_data)
    X = X.join(labels)

    st.header("Normalisasi Data")
    st.write("Split Fitur 'Round' dan Normalisasi Fitur 'Goal Scored', 'Ball Possession %', 'Attempts', 'On-Target', 'Off-Target', 'Offsides', 'Saves', 'Pass Accuracy %', 'Passes', 'Distance Covered (Kms)', 'Fouls Committed', 'Own goals'")

    st.write(X)

    percent_amount_of_test_data = 0.3
    X1 = X.iloc[:,0:18].values
    Y1 = X.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size = percent_amount_of_test_data, random_state=0)

with tab3:
    st.header("Implementasi Aplikasi")
    st.write('Silahkan isi input dibawah ini dengan benar. Setelah itu tekan tombol "Predict" untuk memprediksi')
    with st.form(key ='Form1'):
        goal = st.number_input("Goal Scored", min_value=0)   
        possession = st.number_input("Ball Possession (%)", min_value=0, max_value=100)   
        attempt = st.number_input("Attempts", min_value=0)   
        onTarget = st.number_input("Attempt On Target", min_value=0)   
        offTarget = st.number_input("Attempt Off Target", min_value=0)   
        offsides = st.number_input("Offsides", min_value=0)
        saves = st.number_input("Saves", min_value=0)
        passAcc = st.number_input("Pass Accuracy (%)", min_value=0, max_value=100)
        passMade = st.number_input("Passes Made", min_value=0)
        distance = st.number_input("Distance Covered (Kms)", min_value=0)
        fouls = st.number_input("Fouls Committed", min_value=0)
        rounds = st.selectbox('Round', ('Group Stage', 'Round of 16', 'Quarter Finals', 'Semi- Finals', '3rd Place', 'Final'))
        ownGoal = st.number_input("Own Goal", min_value=0)
        submitted1 = st.form_submit_button(label = 'Predict')
    
    if submitted1:
        st.subheader('Hasil Prediksi')

        st.write()
        
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
        gaussian = joblib.load("nb.joblib")
        knn = joblib.load("knn.joblib")
        tree = joblib.load("tree.joblib")
        y_pred = tree.predict(X_test)
        y_pred1 = knn.predict(X_test)
        y_pred2 = gaussian.predict(X_test)

        col1, col2, col3 = st.columns(3)

        with col1:
            pred2 = gaussian.predict(inputs)
            if pred2[0] == 1:
                st.write("Naive Bayes: Man of the Match")
            else:
                st.write("Naive Bayes: Not Man of the Match")
            st.write("Accuracy Naive Bayes: ", round(100 * accuracy_score(y_test, y_pred2), 2), " %")

        with col2:
            
            pred1 = knn.predict(inputs)
            if pred1[0] == 1:
                st.write("KNN: Man of the Match")
            else:
                st.write("KNN: Not Man of the Match")
            st.write("Accuracy KNN: ", round(100 * accuracy_score(y_test, y_pred1), 2), " %")

        with col3:
            pred = tree.predict(inputs)
            if pred[0] == 1:
                st.write("Tree: Man of the Match")
            else:
                st.write("Tree: Not Man of the Match")
            st.write("Accuracy Tree: ", round(100 * accuracy_score(y_test, y_pred), 2)," %")
        