import warnings

import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')


def diabetes(request):
    return render(request, 'diabetes/diabetes.html')


def predict(request):
    try:
        Pregnancies = request.POST.get("Pregnancies")
        Glucose = request.POST.get("Glucose")
        BloodPressure = request.POST.get("BloodPressure")
        SkinThickness = request.POST.get("SkinThickness")
        Insulin = request.POST.get("Insulin")
        BMI = request.POST.get("BMI")
        DiabetesPedigreeFunction = request.POST.get("DiabetesPedigreeFunction")

        user_response = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction]
        data = {'user_data': user_response}
    except:
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction = None, None, None, None, None, None, None

    diabetes_dataset = pd.read_csv('E:\sem7\SGP\SGP-PROJECT\HEALTH_APP\\templates\DATASETS\diabetes.csv')

    print(diabetes_dataset.head())
    print("\n----------------------------------------------------------------------------------------\n")
    print(diabetes_dataset.shape)
    print("\n----------------------------------------------------------------------------------------\n")
    print(diabetes_dataset.describe())
    print("\n----------------------------------------------------------------------------------------\n")
    print(diabetes_dataset['Outcome'].value_counts())
    print("\n----------------------------------------------------------------------------------------\n")
    print(diabetes_dataset.groupby('Outcome').mean())

    # separating the data and labels
    X = diabetes_dataset[
        ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']]
    Y = diabetes_dataset['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=43)

    classifier = svm.SVC(kernel='linear', C=2)

    # Training the SVM Classifier
    classifier.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    y_pred_svm = classifier.predict([user_response])
    print(y_pred_svm)
    score_svm = classifier.score(X_test, Y_test)*100
    print("svm: accuracy", score_svm)

    if y_pred_svm == 0:
        result_of_SVM = "you are not a heart disease patient"
    else:
        result_of_SVM = "you have symptoms of heart disease"

    # ------------------------------------------------------------------------------------------

    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    y_pred_lr = classifier.predict([user_response])
    print(y_pred_lr)
    score_lr = classifier.score(X_test, Y_test) * 100
    print("Logistic regress~ion accuracy:", score_lr)

    if y_pred_lr[0] == 0:
        result_of_LR = "you are not a heart disease patient"
    else:
        result_of_LR = "you have symptoms of heart disease"

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
        result_of_DT = "you are not a heart disease patient"
    else:
        result_of_DT = "you have symptoms of heart disease"

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
        result_of_NB = "you are not a heart disease patient"
    else:
        result_of_NB = "you have symptoms of heart disease"

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
        result_of_RF = "you are not a heart disease patient"
    else:
        result_of_RF = "you have symptoms of heart disease"

    final_data = {"Support Vector Machine": [score_svm, result_of_SVM],
                  "Logistic Regression": [score_lr, result_of_LR],
                  "Decision Tree": [score_dt, result_of_DT],
                  "Naive Byes": [score_nb, result_of_NB],
                  "Random Forest": [score_rf, result_of_RF]
                  }

    Parameters = {'Pregnancies': user_response[0], 'Glucose': user_response[1], 'BloodPressure': user_response[2],
              'SkinThickness': user_response[3], 'Insulin': user_response[4], 'BMI': user_response[5],
              'DiabetesPedigreeFunction': user_response[6]}

    normal_DiabetesPedigree_pregnency_count = diabetes_dataset[(diabetes_dataset['DiabetesPedigreeFunction'] <= 0.80) &
    (diabetes_dataset['Gender'] == 1)]['Pregnancies'].count()
    high_DiabetesPedigree_pregnency_count = diabetes_dataset[(diabetes_dataset['DiabetesPedigreeFunction'] > 0.80) &
    (diabetes_dataset['Gender'] == 1)]['Pregnancies'].count()

    labels1 = [['High DiabetesPedigree (> 0.80)'], ['Normal DiabetesPedigree (<= 0.80)']]
    data1 = [normal_DiabetesPedigree_pregnency_count, high_DiabetesPedigree_pregnency_count]

    labels2 = [["Support", "Vector Machine"], ["Logistic", "Regression"],
        ["Decision", "Tree"], ["Naive", "Byes"], ["Random", "Forest"]]
    data2 = [score_svm, score_lr, score_dt, score_nb, score_rf]

    Non_diabetic_patient_count = diabetes_dataset[diabetes_dataset['Outcome'] == 0]['Outcome'].count()
    diabetic_patient_count = diabetes_dataset[diabetes_dataset['Outcome'] == 1]['Outcome'].count()

    labels3 = [['Non Diabetic '], ['Diabetic']]
    data3 = [Non_diabetic_patient_count, diabetic_patient_count]

    normal_Diabetic_female_count = diabetes_dataset.loc[(diabetes_dataset['Outcome'] == 0) &
                                                        (diabetes_dataset['Gender'] == 1)]['Gender'].count()

    normal_Diabetic_male_count = diabetes_dataset.loc[(diabetes_dataset['Outcome'] == 0) &
                                                      (diabetes_dataset['Gender'] == 0)]['Gender'].count()

    high_Diabetic_female_count = diabetes_dataset.loc[(diabetes_dataset['Outcome'] == 1) &
                                                      (diabetes_dataset['Gender'] == 1)]['Gender'].count()

    high_Diabetic_male_count = diabetes_dataset.loc[(diabetes_dataset['Outcome'] == 1) &
                                                    (diabetes_dataset['Gender'] == 0)]['Gender'].count()

    labels4 = [["Non diabetic", "Female Patient"], ["Non diabetic", "Male Patient"],
        ["diabetic", "Female Patient"], ["diabetic", "Male Patient"]]

    data4 = [normal_Diabetic_female_count, normal_Diabetic_male_count, high_Diabetic_female_count, high_Diabetic_male_count]

    High_BMI_female = diabetes_dataset[(diabetes_dataset['BMI'] > 25) & (diabetes_dataset['Gender'] == 1)]['Gender'].count()
    High_BMI_male = diabetes_dataset[(diabetes_dataset['BMI'] > 25) & (diabetes_dataset['Gender'] == 0)]['Gender'].count()
    normal_BMI_female = diabetes_dataset[(diabetes_dataset['BMI'] <= 25) & (diabetes_dataset['Gender'] == 1)]['Gender'].count()
    normal_BMI_male = diabetes_dataset[(diabetes_dataset['BMI'] <= 25) & (diabetes_dataset['Gender'] == 0)]['Gender'].count()

    labels5 = ["High BMI (Female)", "High BMI (Male)", "Normal BMI (female)", "(Normal BMI Male)"]
    data5 = [High_BMI_female, High_BMI_male, normal_BMI_female, normal_BMI_male]

    High_glucose_female = diabetes_dataset[(diabetes_dataset['Glucose'] > 110) & (diabetes_dataset['Gender'] == 1)]['Gender'].count()
    High_glucose_male = diabetes_dataset[(diabetes_dataset['Glucose'] > 110) & (diabetes_dataset['Gender'] == 0)]['Gender'].count()
    normal_glucose_female = diabetes_dataset[(diabetes_dataset['Glucose'] <= 110) & (diabetes_dataset['Gender'] == 1)]['Gender'].count()
    normal_glucose_male = diabetes_dataset[(diabetes_dataset['Glucose'] <= 100) & (diabetes_dataset['Gender'] == 0)]['Gender'].count()

    labels6 = [["High Glucose", "(Female)"], ["High Glucose", "(Male)"] , ["Normal Glucose", "(Female)"], ["Normal Glucose", "(Male)"]]
    data6 = [High_glucose_female, High_glucose_male, normal_glucose_female, normal_glucose_male]

    High_Bloodpressure_female = diabetes_dataset[(diabetes_dataset['BloodPressure'] > 110) & (diabetes_dataset['Gender'] == 1)][
        'Gender'].count()
    High_Bloodpressure_male = diabetes_dataset[(diabetes_dataset['BloodPressure'] > 110) & (diabetes_dataset['Gender'] == 0)][
        'Gender'].count()
    normal_Bloodpressure_female = diabetes_dataset[(diabetes_dataset['BloodPressure'] <= 110) & (diabetes_dataset['Gender'] == 1)][
        'Gender'].count()
    normal_Bloodpressure_male = diabetes_dataset[(diabetes_dataset['BloodPressure'] <= 100) & (diabetes_dataset['Gender'] == 0)][
        'Gender'].count()

    labels7 = [["High Bloodpressure", "(Female)"], ["High Bloodpressure", "(Male)"], ["Normal Bloodpressure", "(Female)"],
               ["Normal Bloodpressure", "(Male)"]]
    data7 = [High_Bloodpressure_female, High_Bloodpressure_male, normal_Bloodpressure_female, normal_Bloodpressure_male]

    return render(request, 'diabetes/result.html',
                  {'Parameters': Parameters, "data": final_data, "labels1": labels1, "data1": data1, "labels2": labels2, "data2": data2,
                   "labels3": labels3, "data3": data3, "labels4": labels4, "data4": data4, "labels5": labels5, "data5": data5,
                   "labels6": labels6, "data6": data6, "labels7": labels7, "data7": data7})



