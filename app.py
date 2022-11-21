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

st.header('Prediksi Resiko Kredit Customer')

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




# def modelData(overdue_0_30, overdue_31_45, overdue_46_60, overdue_61_90, overdue_90, kpr_tidak, kpr_aktif, pendapatan, durasi, tanggungan):
#      #Mengubah Fitur "rata_rata_overdue" Menjadi Tipe Data Numerik

#     data = load_data()

#     labels = data["risk_rating"]
#     X = data.drop(columns=['risk_rating'])

#     split_overdue_X = pd.get_dummies(X["rata_rata_overdue"], prefix="overdue")
#     X = X.join(split_overdue_X)

#     X = X.drop(columns = "rata_rata_overdue")

#     # Normalisasi Data

#     KPR_status = pd.get_dummies(X["kpr_aktif"], prefix="KPR")
#     X = X.join(KPR_status)

#     # remove "rata_rata_overdue" feature
#     X = X.drop(columns = "kpr_aktif")

#     old_normalize_feature_labels = ['pendapatan_setahun_juta', 'durasi_pinjaman_bulan', 'jumlah_tanggungan']
#     new_normalized_feature_labels = ['norm_pendapatan_setahun_juta', 'norm_durasi_pinjaman_bulan', 'norm_jumlah_tanggungan']
#     normalize_feature = data[old_normalize_feature_labels]

#     scaler = MinMaxScaler()
#     scaler.fit(normalize_feature)

#     normalized_feature = scaler.transform(normalize_feature)
#     normalized_feature_data = pd.DataFrame(normalized_feature, columns = new_normalized_feature_labels)

#     X = X.drop(columns = old_normalize_feature_labels)
#     X = X.join(normalized_feature_data)
#     X = X.join(labels)

#     subject_lables = ["Unnamed: 0",  "kode_kontrak"]
#     X = X.drop(columns = subject_lables)
#     # percent_amount_of_test_data = / HUNDRED_PERCENT
#     percent_amount_of_test_data = 0.3

#     # Hitung data

#     # values
#     matrices_X = X.iloc[:,0:10].values

#     # classes
#     matrices_Y = X.iloc[:,10].values

#     X_1 = X.iloc[:,0:10].values
#     Y_1 = X.iloc[:, -1].values
#     X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size = percent_amount_of_test_data, random_state=0)


#     # Implementasi gaussian naive bayes

#     gaussian = GaussianNB()
#     gaussian.fit(X_train, y_train)
#     Y_pred = gaussian.predict(X_test) 
#     accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
#     acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

#     cm = confusion_matrix(y_test, Y_pred)
#     accuracy = accuracy_score(y_test,Y_pred)
#     precision =precision_score(y_test, Y_pred,average='micro')
#     recall =  recall_score(y_test, Y_pred,average='micro')
#     f1 = f1_score(y_test,Y_pred,average='micro')
#     st.write('Confusion matrix for Naive Bayes\n',cm)
#     st.write('accuracy_Naive Bayes: %.3f' %accuracy)
#     st.write('precision_Naive Bayes: %.3f' %precision)
#     st.write('recall_Naive Bayes: %.3f' %recall)
#     st.write('f1-score_Naive Bayes : %.3f' %f1)


#     clf = GaussianNB()
#     clf.fit(matrices_X, matrices_Y)
#     clf_pf = GaussianNB()
#     clf_pf.partial_fit(matrices_X, matrices_Y, np.unique(matrices_Y))

#     result_test_naive_bayes = clf_pf.predict([
#         [overdue_0_30, overdue_31_45, overdue_46_60, overdue_61_90, overdue_90 ,kpr_tidak, kpr_aktif, pendapatan, durasi, tanggungan]
#         ])[0]

#     return(result_test_naive_bayes)


if submitted1:
    
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