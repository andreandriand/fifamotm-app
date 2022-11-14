import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import plotly.express as px

### Modelling 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

### Remove unnecessary warnings
import warnings
warnings.filterwarnings('ignore')


px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = 'reds'

st.header('Credit Scoring')

def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/andreandriand/dataset/main/credit_score.csv')
    return df

data = load_data()


labels = data["risk_rating"]
X = data.drop(columns=['risk_rating'])

st.subheader('Dataset Credit Scoring')

st.write(data)


# Sidebar

st.sidebar.header('User Input Features')

with st.form(key ='Form1'):
    with st.sidebar:
        pendapatan = st.sidebar.number_input("Pendapatan Per Tahun", min_value=0, max_value=10000)   
        tanggungan = st.sidebar.number_input("Jumlah Tanggungan", min_value=0)   
        kpr = st.sidebar.radio('KPR', ('Aktif', 'Tidak'))
        durasi = st.sidebar.selectbox('Durasi Pinjaman', ('12 Bulan', '24 Bulan', '36 Bulan', '48 Bulan'))
        overdues = st.sidebar.radio('Rata - Rata Keterlambatan', ('0-30 Hari', '31-45 Hari', '46-60 Hari', '61-90 Hari', '>90 Hari'))
        submitted1 = st.form_submit_button(label = 'Kalkulasi')




def modelData(overdue_0_30, overdue_31_45, overdue_46_60, overdue_61_90, overdue_90, kpr_tidak, kpr_aktif, pendapatan, durasi, tanggungan):
     #Mengubah Fitur "rata_rata_overdue" Menjadi Tipe Data Numerik

    data = load_data()

    labels = data["risk_rating"]
    X = data.drop(columns=['risk_rating'])

    split_overdue_X = pd.get_dummies(X["rata_rata_overdue"], prefix="overdue")
    X = X.join(split_overdue_X)

    X = X.drop(columns = "rata_rata_overdue")

    # Normalisasi Data

    KPR_status = pd.get_dummies(X["kpr_aktif"], prefix="KPR")
    X = X.join(KPR_status)

    # remove "rata_rata_overdue" feature
    X = X.drop(columns = "kpr_aktif")

    old_normalize_feature_labels = ['pendapatan_setahun_juta', 'durasi_pinjaman_bulan', 'jumlah_tanggungan']
    new_normalized_feature_labels = ['norm_pendapatan_setahun_juta', 'norm_durasi_pinjaman_bulan', 'norm_jumlah_tanggungan']
    normalize_feature = data[old_normalize_feature_labels]

    scaler = MinMaxScaler()
    scaler.fit(normalize_feature)

    normalized_feature = scaler.transform(normalize_feature)
    normalized_feature_data = pd.DataFrame(normalized_feature, columns = new_normalized_feature_labels)

    X = X.drop(columns = old_normalize_feature_labels)
    X = X.join(normalized_feature_data)
    X = X.join(labels)

    subject_lables = ["Unnamed: 0",  "kode_kontrak"]
    X = X.drop(columns = subject_lables)
    # percent_amount_of_test_data = / HUNDRED_PERCENT
    percent_amount_of_test_data = 0.3

    # Hitung data

    # values
    matrices_X = X.iloc[:,0:10].values

    # classes
    matrices_Y = X.iloc[:,10].values

    X_1 = X.iloc[:,0:10].values
    Y_1 = X.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size = percent_amount_of_test_data, random_state=0)


    # Implementasi gaussian naive bayes

    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    # st.write('Confusion matrix for Naive Bayes\n',cm)
    # st.write('accuracy_Naive Bayes: %.3f' %accuracy)
    # st.write('precision_Naive Bayes: %.3f' %precision)
    # st.write('recall_Naive Bayes: %.3f' %recall)
    # st.write('f1-score_Naive Bayes : %.3f' %f1)


    clf = GaussianNB()
    clf.fit(matrices_X, matrices_Y)
    clf_pf = GaussianNB()
    clf_pf.partial_fit(matrices_X, matrices_Y, np.unique(matrices_Y))

    result_test_naive_bayes = clf_pf.predict([
        [overdue_0_30, overdue_31_45, overdue_46_60, overdue_61_90, overdue_90 ,kpr_tidak, kpr_aktif, pendapatan, durasi, tanggungan]
        ])[0]

    return(result_test_naive_bayes)


if submitted1:
    kpr_aktif = 0
    kpr_tidak = 0
    if kpr == 'Aktif':
        kpr_aktif = 1
    else:
        kpr_tidak = 1

    if durasi == '12 Bulan':
        durasi = 12
    elif durasi == '24 Bulan':
        durasi = 24
    elif durasi == '36 Bulan':
        durasi = 36
    else:
        durasi = 48

    overdue = [0,0,0,0,0]
    if overdues == '0-30 Hari':
        overdue[0] = 1
    elif overdues == '31-45 Hari':
        overdue[1] = 1
    elif overdues == '46-60 Hari':
        overdue[2] = 1
    elif overdues == '61-90 Hari':
        overdue[3] = 1
    else:
        overdue[4] = 1


    inputs = {
        'pendapatan': pendapatan,
        'tanggungan': tanggungan,
        'durasi': durasi,
        'kpr_aktif': kpr_aktif,
        'kpr_tidak': kpr_tidak,
        'overdue_0-30': overdue[0],
        'overdue_31-45': overdue[1],
        'overdue_46-60': overdue[2],
        'overdue_61-90': overdue[3],
        'overdue_>90': overdue[4]
    }

    hitung = modelData(inputs['overdue_0-30'], inputs['overdue_31-45'], inputs['overdue_46-60'], inputs['overdue_61-90'], inputs['overdue_>90'], inputs['kpr_tidak'], inputs['kpr_aktif'], inputs['pendapatan'], inputs['durasi'], inputs['tanggungan'])

    st.subheader('Hasil Prediksi')

    st.write("Customer dengan data :")

    st.write("Pendapatan : ", pendapatan)
    st.write("Durasi Pinjaman : ", durasi)
    st.write("Jumlah Tanggungan : ", tanggungan)
    st.write("KPR : ", kpr)
    st.write("Overdue : ", overdues)

    st.write("Memiliki resiko sebesar : ",hitung, "berdasarkan model Gaussian Naive Bayes")