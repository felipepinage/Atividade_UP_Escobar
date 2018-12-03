import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib

# RESULTADOS
def resultados(labelsTeste, predicoes):
    acc = metrics.accuracy_score(labelsTeste, predicoes)
    fscore = metrics.f1_score(labelsTeste, predicoes, average='weighted')
    prec = metrics.precision_score(labelsTeste, predicoes, average='weighted')
    recall = metrics.recall_score(labelsTeste, predicoes, average='weighted')
    print("EVALUATION METRICS\n"
          "acc: %0.4f - fscore: %0.4f - prec: %0.4f - recall: %0.4f\n" % (acc, fscore, prec, recall))
    return [acc, fscore, prec, recall]


df = pd.read_csv('Absenteeism_at_work.csv', sep=';', header=0).replace(np.NaN, 0)

data = df.drop(['Reason for absence'], axis=1) #separate data from labels
lbl = np.where(df['Reason for absence'] <= 21, 1, 0) #1: CID; 0: non-CID

data = data.div(data.sum(axis=1), axis=0) #normalizate data to build the classifier
data.to_csv("usr_input.csv") #Save processed data to a new dataset

dataNoLabels = np.asmatrix(data) #data: dataframe to nArray

X_train, X_test, y_train, y_test = train_test_split(dataNoLabels, lbl, test_size=0.15, random_state=0) #split the dataset to train and test
clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_train, np.ravel(y_train,order='C')) #train RandomForest
predicoes = clf.predict(X_test) #test RandomForest
print('Fase de Treino/Teste')
acc, f1s, precisao, recall = resultados(y_test, predicoes) #show results

#--------------------------------------------------------#

joblib.dump(clf, 'RF_up.joblib') #save the new model (limited) as joblib file
clf = joblib.load('RF_up.joblib')
usr_input = pd.read_csv('usr_input.csv', sep=',', header=0, usecols= range(1,21)).replace(np.NaN, 0)
usr_input2 = np.asmatrix(usr_input) #data: dataframe to nArray
simulacao = clf.predict(usr_input)
print('Fase de execução com Usuário')
acc3, f1s3, precisao3, recall3 = resultados(lbl, simulacao) #show results