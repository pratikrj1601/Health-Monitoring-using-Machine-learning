import base64
import io
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.shortcuts import render
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def liver(request):
    return render(request, 'liver/liver.html')


def predict(request):
    try:

        Total_Bilirubin = request.POST.get("Total_Bilirubin")
        Direct_Bilirubin = request.POST.get("Direct_Bilirubin")
        Alkaline_Phosphotase = request.POST.get("Alkaline_Phosphotase")
        Alamine_Aminotransferase = request.POST.get("Alamine_Aminotransferase")
        Aspartate_Aminotransferase = request.POST.get("Aspartate_Aminotransferase")
        Total_Protiens = request.POST.get("Total_Protiens")
        Albumin = request.POST.get("Albumin")
        Albumin_and_Globulin_Ratio = request.POST.get("Albumin_and_Globulin_Ratio")

        user_response = [Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,
                         Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]

        data = {'user_data': user_response}

    except:
        Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio = None, None, None, None, None, None, None, None

    liver_dataset = pd.read_csv('E:\sem7\SGP\SGP-PROJECT\HEALTH_APP\\templates\DATASETS\liver.csv')
    liver_dataset.dropna()

    print(liver_dataset.head())
    print("\n----------------------------------------------------------------------------------------\n")
    print(liver_dataset.shape)
    print("\n----------------------------------------------------------------------------------------\n")
    print(liver_dataset.describe())
    print("\n----------------------------------------------------------------------------------------\n")
    print(liver_dataset['Result'].value_counts())
    print("\n----------------------------------------------------------------------------------------\n")
    print(liver_dataset.groupby('Result').mean())

    X = liver_dataset[['Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                       'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
    Y = liver_dataset['Result']

    # ============================================================================================================================

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=43)

    classifier = svm.SVC(kernel='linear', C=2)

    # Training the SVM Classifier
    classifier.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    y_pred_svm = classifier.predict([user_response])
    print(y_pred_svm)
    score_svm = classifier.score(X_test, Y_test) * 100
    print("svm: accuracy", score_svm)

    if y_pred_svm == 0:
        result_of_SVM = "you are not a liver disease patient"
    else:
        result_of_SVM = "you have symptoms of liver disease"

    # ------------------------------------------------------------------------------------------

    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    y_pred_lr = classifier.predict([user_response])
    print(y_pred_lr)
    score_lr = classifier.score(X_test, Y_test) * 100
    print("Logistic regression accuracy:", score_lr)

    if y_pred_lr[0] == 0:
        result_of_LR = "you are not a liver disease patient"
    else:
        result_of_LR = "you have symptoms of liver disease"

    # ------------------------------------------------------------------------------------------

    classifier_dt = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    classifier_dt.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    # Predicting the test set result
    y_pred_dt = classifier_dt.predict([user_response])
    print(y_pred_dt)
    score_dt = classifier_dt.score(X_test, Y_test) * 100
    print("Decision Tree accuracy:", score_dt)

    if y_pred_dt[0] == 0:
        result_of_DT = "you are not a liver disease patient"
    else:
        result_of_DT = "you have symptoms of liver disease"

    # -----------------------------------------------------------------------------------------------------------------

    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    # making predictions on the testing set
    y_pred_nb = gnb.predict([user_response])
    print(y_pred_nb)

    # comparing actual response values (Y_test) with predicted response values (y_pred)

    score_nb = gnb.score(X_test, Y_test) * 100
    print("Naive Byes accuracy:", score_nb)

    if y_pred_nb[0] == 0:
        result_of_NB = "you are not a liver disease patient"
    else:
        result_of_NB = "you have symptoms of liver disease"

    # -----------------------------------------------------------------------------------------------------------------

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    # making predictions on the testing set
    y_pred_rf = rf.predict([user_response])
    print(y_pred_rf)

    # comparing actual response values (Y_test) with predicted response values (y_pred)

    score_rf = rf.score(X_test, Y_test) * 100
    print("RF accuracy:", score_rf)

    if (y_pred_rf[0] == 0):
        result_of_RF = "you are not a liver disease patient"
    else:
        result_of_RF = "you have symptoms of liver disease"

    final_data = {"Support Vector Machine": [score_svm, result_of_SVM],
                  "Logistic Regression": [score_lr, result_of_LR],
                  "Decision Tree": [score_dt, result_of_DT],
                  "Naive Byes": [score_nb, result_of_NB],
                  "Random Forest": [score_rf, result_of_RF]}

    parameters = {'Total_Bilirubin': user_response[0], 'Direct_Bilirubin': user_response[1], 'Alkaline_Phosphotase': user_response[2],
                 'Alamine_Aminotransferase': user_response[3], 'Total_Protiens': user_response[4],
                 'Albumin': user_response[5], 'Albumin_and_Globulin_Ratio': user_response[6] }

    labels1 = [["Support", "Vector Machine"], ["Logistic", "Regression"],
               ["Decision", "Tree"], ["Naive", "Byes"], ["Random", "Forest"]]
    data1 = [score_svm, score_lr, score_dt, score_nb, score_rf]

    Non_liver_patient_count = liver_dataset[liver_dataset['Result'] == 0]['Result'].count()
    liver_patient_count = liver_dataset[liver_dataset['Result'] == 1]['Result'].count()

    labels2 = [['Not a liver patient '], ['liver patient']]
    data2 = [Non_liver_patient_count, liver_patient_count]

    normal_liver_female_count = liver_dataset.loc[(liver_dataset['Result'] == 0) &
                                                        (liver_dataset['Gender'] == 1)]['Gender'].count()

    normal_liver_male_count = liver_dataset.loc[(liver_dataset['Result'] == 0) &
                                                      (liver_dataset['Gender'] == 0)]['Gender'].count()

    high_liver_female_count = liver_dataset.loc[(liver_dataset['Result'] == 1) &
                                                      (liver_dataset['Gender'] == 1)]['Gender'].count()

    high_liver_male_count = liver_dataset.loc[(liver_dataset['Result'] == 1) &
                                                    (liver_dataset['Gender'] == 0)]['Gender'].count()

    labels3 = [["Non liver", "Female Patient"], ["Non liver", "Male Patient"],
               ["liver", "Female Patient"], ["liver", "Male Patient"]]

    data3 = [normal_liver_female_count, normal_liver_male_count, high_liver_female_count,
             high_liver_male_count]

    normal_Total_Protiens_female_count = liver_dataset[(liver_dataset['Total_Protiens'] <= 8.3) &
                                                               (liver_dataset['Gender'] == 1)]['Gender'].count()
    high_Total_Protiens_female_count = liver_dataset[(liver_dataset['Total_Protiens'] > 8.3) &
                                                             (liver_dataset['Gender'] == 1)]['Gender'].count()
    normal_Total_Protiens_male_count = liver_dataset[(liver_dataset['Total_Protiens'] <= 8.3) &
                                                       (liver_dataset['Gender'] == 0)]['Gender'].count()
    high_Total_Protiens_male_count = liver_dataset[(liver_dataset['Total_Protiens'] > 8.3) &
                                                     (liver_dataset['Gender'] == 0)]['Gender'].count()

    labels4 = [['Normal Protiens', 'Female'], ['High Protiens', 'Female'],
               ['Normal Protiens', 'male'], ['High Protiens', 'male']]
    data4 = [normal_Total_Protiens_female_count, high_Total_Protiens_female_count,
             normal_Total_Protiens_male_count, high_Total_Protiens_male_count]

    low_bilirubin_patient_count = liver_dataset[liver_dataset['Total_Bilirubin'] <= 1.9]['Result'].count()
    high_bilirubin_patient_count = liver_dataset[liver_dataset['Total_Bilirubin'] > 1.9]['Result'].count()

    labels5 = [['low bilirubin patient '], ['high bilirubin patient']]
    data5 = [low_bilirubin_patient_count, high_bilirubin_patient_count]

    normal_Albumin_and_Globulin_Ratio_female_count = liver_dataset[(liver_dataset['Albumin_and_Globulin_Ratio']
                                        <= 2.5) & (liver_dataset['Gender'] == 1)]['Gender'].count()
    high_Albumin_and_Globulin_Ratio_female_count = liver_dataset[(liver_dataset['Albumin_and_Globulin_Ratio'] > 2.5) &
                                                     (liver_dataset['Gender'] == 1)]['Gender'].count()
    normal_Albumin_and_Globulin_Ratio_male_count = liver_dataset[(liver_dataset['Albumin_and_Globulin_Ratio'] <= 2.5) &
                                                     (liver_dataset['Gender'] == 0)]['Gender'].count()
    high_Albumin_and_Globulin_Ratio_male_count = liver_dataset[(liver_dataset['Albumin_and_Globulin_Ratio'] > 2.5) &
                                                   (liver_dataset['Gender'] == 0)]['Gender'].count()

    labels6 = [['Normal Albumin & Globulin Ratio Female'], ['High Albumin & Globulin Ratio female'],
               ['Normal Albumin & Globulin Ratio male'], ['High Albumin & Globulin Ratio male']]
    data6 = [normal_Albumin_and_Globulin_Ratio_female_count, high_Albumin_and_Globulin_Ratio_female_count,
             normal_Albumin_and_Globulin_Ratio_male_count, high_Albumin_and_Globulin_Ratio_male_count]

    return render(request, 'liver/result.html', {'parameters': parameters, "data": final_data, 'labels1': labels1,
                                                 'data1': data1, "labels2": labels2, "data2": data2, "labels3":
                                                 labels3, "data3": data3, "labels4": labels4, "data4": data4,
                                                 "labels5": labels5, "data5": data5, "labels6": labels6,
                                                 "data6": data6})
