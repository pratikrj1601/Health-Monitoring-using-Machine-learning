import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Create your views here.
from sklearn.tree import DecisionTreeClassifier


def cancer(request):

    return render(request, 'CANCER/cancer.html')


def predict(request):

    try:
        Clump_Thickness = request.POST.get("Clump_Thickness")
        Cell_Size_Uniformity = request.POST.get("Cell_Size_Uniformity")
        Cell_Shape_Uniformity = request.POST.get("Cell_Shape_Uniformity")
        Marginal_Adhesion = request.POST.get("Marginal_Adhesion")
        Single_Epi_Cell_Size = request.POST.get("Single_Epi_Cell_Size")
        Bare_Nuclei = request.POST.get("Bare_Nuclei")
        Bland_Chromatin = request.POST.get("Bland_Chromatin")
        Normal_Nucleoli = request.POST.get("Normal_Nucleoli")
        Mitoses = request.POST.get("Mitoses")

        user_response = [Clump_Thickness, Cell_Size_Uniformity, Cell_Shape_Uniformity, Marginal_Adhesion, Single_Epi_Cell_Size,
                         Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses]

    except:
        Clump_Thickness, Cell_Size_Uniformity, Cell_Shape_Uniformity, Marginal_Adhesion, Single_Epi_Cell_Size,
        Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses = None, None, None, None, None, None, None

    cancer_df = pd.read_csv("E:\sem7\SGP\SGP-PROJECT\HEALTH_APP\\templates\DATASETS\\breast_cancer_dataset.csv")

    print(cancer_df.head())
    print("\n----------------------------------------------------------------------------------------\n")
    print(cancer_df.shape)
    print("\n----------------------------------------------------------------------------------------\n")
    print(cancer_df.describe())
    print("\n----------------------------------------------------------------------------------------\n")
    print(cancer_df['Class'].value_counts())
    print("\n----------------------------------------------------------------------------------------\n")
    print(cancer_df.groupby('Class').mean())

    # separating the data and labels
    X = cancer_df[['Clump_Thickness', 'Cell_Size_Uniformity', 'Cell_Shape_Uniformity', 'Marginal_Adhesion', 'Single_Epi_Cell_Size',
                         'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses']]
    Y = cancer_df['Class']

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
        result_of_SVM = "you are not a cancer disease patient"
    else:
        result_of_SVM = "you have symptoms of cancer disease"

    # ------------------------------------------------------------------------------------------

    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)

    np.reshape(user_response, [1, -1])
    y_pred_lr = classifier.predict([user_response])
    print(y_pred_lr)
    score_lr = classifier.score(X_test, Y_test) * 100
    print("Logistic regression accuracy:", score_lr)

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
    y_pred_nb = gnb.predict([user_response])
    print(y_pred_nb)

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
    y_pred_rf = rf.predict([user_response])
    print(y_pred_rf)

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

    Parameters = {'Clump Thickness': user_response[0], 'Cell Size Uniformity': user_response[1],
                  'Cell Shape Uniformity': user_response[2], 'Marginal Adhesion': user_response[3],
                  'Single Epi Cell Size': user_response[4], 'Bare Nuclei': user_response[5],
                  'Bland Chromatin': user_response[6], 'Normal Nucleoli': user_response[7], 'Mitoses': user_response[8]}

    labels1 = [["Support", "Vector Machine"], ["Logistic", "Regression"],
               ["Decision", "Tree"], ["Naive", "Byes"], ["Random", "Forest"]]
    data1 = [score_svm, score_lr, score_dt, score_nb, score_rf]

    high_Clump_Thickness = cancer_df[cancer_df['Class'] == 1]['Clump_Thickness'].count()
    low_Clump_Thickness = cancer_df[cancer_df['Class'] == 0]['Clump_Thickness'].count()

    high_Cell_Size_Uniformity = cancer_df[cancer_df['Class'] == 1]['Cell_Size_Uniformity'].count()
    low_Cell_Size_Uniformity = cancer_df[cancer_df['Class'] == 0]['Cell_Size_Uniformity'].count()

    high_Cell_Shape_Uniformity = cancer_df[cancer_df['Class'] == 1]['Cell_Shape_Uniformity'].count()
    low_Cell_Shape_Uniformity = cancer_df[cancer_df['Class'] == 0]['Cell_Shape_Uniformity'].count()

    labels2 = ['High clump thickess', 'low clump thickess']
    data2 = [high_Clump_Thickness, low_Clump_Thickness]

    labels3 = ['High cell size uniformity', 'low cell size uniformity']
    data3 = [high_Cell_Size_Uniformity, low_Cell_Size_Uniformity]

    labels4 = ['High cell shape uniformity', 'low cell shape uniformity']
    data4 = [high_Cell_Shape_Uniformity, low_Cell_Shape_Uniformity]

    return render(request, 'CANCER/result.html', {'Parameters': Parameters, "data": final_data, "labels1": labels1,
                                                  "data1": data1, "labels2": labels2, "data2": data2,
                                                  "labels3": labels3, "data3": data3, "labels4": labels4,
                                                  "data4": data4})
