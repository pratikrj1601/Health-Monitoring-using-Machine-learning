import base64
import io
import urllib.parse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


# Create your views here.


def heart_disease(request):
    return render(request, 'heart_disease/heart_disease.html')


def predict(request):
    try:
        age = request.POST.get("age")
        sex = request.POST.get("Sex")
        cp = request.POST.get("cp")
        trestbps = request.POST.get("trestbps")
        chol = request.POST.get("chol")
        fbs = request.POST.get("fbs")
        restecg = request.POST.get("restecg")
        thalach = request.POST.get("thalach")
        exang = request.POST.get("exang")
        oldpeak = request.POST.get("oldpeak")
        slope = request.POST.get("slope")
        ca = request.POST.get("ca")
        thal = request.POST.get("thal")

        user_response = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    except:
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = None

    df = pd.read_csv("E:\sem7\SGP\SGP-PROJECT\HEALTH_APP\\templates\DATASETS\heart.csv")
    print(df.head())
    print("\n----------------------------------------------------------------------------------------\n")
    print(df.shape)
    print("\n----------------------------------------------------------------------------------------\n")
    print(df.describe())
    print("\n----------------------------------------------------------------------------------------\n")
    print(df['target'].value_counts())
    print("\n----------------------------------------------------------------------------------------\n")
    print(df.groupby('target').mean())

    # get correlations of each features in dataset
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(12, 9))
    # plot heat map
    g = sns.heatmap(df[top_corr_features].corr(), annot=True, linewidths=.5, cmap="RdYlGn")
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=10)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=10)
    # plt.show()

    fig = g.get_figure()
    # convert graph into string buffer and then we convert 64 bit code into image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    df.hist()

    X = df.drop(columns='target', axis=1)
    Y = df['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=36)

    new_df = pd.DataFrame([user_response],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                                   'oldpeak', 'slope', 'ca', 'thal'])

    classifier = svm.SVC(kernel='linear', C=1, gamma=10)

    # Training the SVM Classifier
    classifier.fit(X_train, Y_train)

    y_pred_svm = classifier.predict(new_df)
    print(y_pred_svm)
    score_svm = classifier.score(X_test, Y_test)
    print("svm: accuracy", score_svm * 100)

    if (y_pred_svm[0] == 0):
        result_of_SVM = "you are not a heart disease patient"
    else:
        result_of_SVM = "you are a heart disease patient"

    # ------------------------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

    # Predict on dataset which model has not seen before
    new_df = pd.DataFrame([user_response],
    columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    classifier = LogisticRegression(random_state=40)
    classifier.fit(X_train, y_train)

    y_pred_lr = classifier.predict(new_df)
    print(y_pred_lr)
    score_lr = classifier.score(X_test, y_test)
    print("Logistic regression accuracy:", score_lr * 100)

    if (y_pred_lr[0] == 0):
        result_of_LR = "you are not a heart disease patient"
    else:
        result_of_LR = "you are a heart disease patient"

    # ------------------------------------------------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)

    # feature Scaling

    classifier_dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=4)
    classifier_dt.fit(x_train, y_train)

    # Predicting the test set result
    y_pred_dt = classifier_dt.predict(new_df)
    print(y_pred_dt)
    score_dt = classifier_dt.score(x_test, y_test)
    print("Decision Tree accuracy:", score_dt * 100)

    if (y_pred_dt[0] == 0):
        result_of_DT = "you are not a heart disease patient"
    else:
        result_of_DT = "you are a heart disease patient"

    # -----------------------------------------------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43)

    # training the model on training set

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_nb = gnb.predict(new_df)
    print(y_pred_nb)


    # comparing actual response values (y_test) with predicted response values (y_pred)

    score_nb = gnb.score(X_test, y_test)
    print("Naive Byes accuracy:", score_nb * 100)

    if (y_pred_nb[0] == 0):
        result_of_NB = "you are not a heart disease patient"
    else:
        result_of_NB = "you are a heart disease patient"

    # -----------------------------------------------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=43)

    # training the model on training set

    rf = RandomForestClassifier(n_estimators = 100)
    rf.fit(X_train, y_train)

    # making predictions on the testing set
    y_pred_rf = rf.predict(new_df)
    print(y_pred_rf)

    # comparing actual response values (y_test) with predicted response values (y_pred)

    score_rf = rf.score(X_test, y_test)
    print("RF accuracy:", score_rf * 100)

    if (y_pred_rf[0] == 0):
        result_of_RF = "you are not a heart disease patient"
    else:
        result_of_RF = "you are a heart disease patient"

    final_data = {"Support Vector Machine": [score_svm*100, result_of_SVM],
                  "Logistic Regression": [score_lr*100, result_of_LR],
                  "Decision Tree": [score_dt*100, result_of_DT],
                  "Naive Byes": [score_nb*100, result_of_NB],
                  "Random Forest": [score_rf*100, result_of_RF]
                  }

    Labels = {}
    if user_response[1] == '0':
        sex_response = 'Female'
    else:
        sex_response = 'Male'

    if user_response[2] == '0':
        cp_response = "Typical Angina"
    elif user_response[2] == '1':
        cp_response = "Atypical Angina"
    elif user_response[2] == '2':
        cp_response = "Non-Anginal Pain"
    elif user_response[2] == '3':
        cp_response = "Asymptomatic"

    if user_response[5] == '0':
        fbs_response = "Fasting Blood Sugar < 120 mg/dl"
    else:
        fbs_response = 'Fasting Blood Sugar > 120 mg/dl'

    if user_response[6] == '0':
        ecg_response = "Normal"
    elif user_response[6] == '1':
        ecg_response = "having ST-T wave abnormality"
    elif user_response[6] == '2':
        ecg_response = "showing probable or definite left ventricular hypertrophy"

    if user_response[8] == '0':
        exang_response = 'No'
    else:
        exang_response = 'Yes'

    if user_response[10] == '0':
        slope_response = "Upslopping"
    elif user_response[10] == '1':
        slope_response = "flat"
    elif user_response[10] == '2':
        slope_response = "Downsloping"

    if user_response[11] == '0':
        ca_response = "0-10 = calcium detected in extremely minimal levels"
    elif user_response[11] == '1':
        ca_response = "11-100 = mild levels of plaque detected with certainty"
    elif user_response[11] == '2':
        ca_response = "101-300 = moderate levels of plaque detected"
    elif user_response[11] == '3':
        ca_response = "300-400 = extensive levels of plaque detected"

    if user_response[12] == '1':
        thal_response = "3 = normal"
    elif user_response[12] == '2':
        thal_response = "6 = fixed defect"
    elif user_response[12] == '3':
        thal_response = "7 = reversable defect"

    Labels = {'age': user_response[0], 'sex': sex_response, 'chest pain': cp_response, 'resting blood pressure': user_response[3],
              'cholestrol': user_response[4], 'fasting blood sugar': fbs_response, 'Electro-cardiographic Result': ecg_response,
              'Maximum Heart Rate Achieved': user_response[7], 'Exercise Induced Angina': exang_response,
              'ST depression induced by exercise': user_response[9], 'slope': slope_response, 'Calcium Heart Score': ca_response,
              'Thalassemia': thal_response}

    Non_heart_disease_patient_count = df[df['target'] == 0]['target'].count()
    heart_disease_patient_count = df[df['target'] == 1]['target'].count()

    labels1 = ['Not a heart disease patient', 'heart disease patient']
    data1 = [Non_heart_disease_patient_count, heart_disease_patient_count]

    return render(request, 'heart_disease/result.html', {'URI': uri, 'labels': Labels, "data": final_data, "labels1": labels1, "data1": data1})