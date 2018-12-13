import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib

# OVERSAMPLING
def oversampling(X_dados, y_labels):
    rus = SMOTE(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_dados, np.ravel(y_labels, order='C'))
    return [X_resampled, y_resampled]

# RESULTADOS
def resultados(labelsTeste, predicoes):
    acc = metrics.accuracy_score(labelsTeste, predicoes)
    fscore = metrics.f1_score(labelsTeste, predicoes, average='weighted')
    prec = metrics.precision_score(labelsTeste, predicoes, average='weighted')
    recall = metrics.recall_score(labelsTeste, predicoes, average='weighted')
    print("EVALUATION METRICS\n"
          "acc: %0.4f - fscore: %0.4f - prec: %0.4f - recall: %0.4f\n" % (acc, fscore, prec, recall))
    return [acc, fscore, prec, recall]


df = pd.read_csv('breast_cancer.csv', sep=',', header=0)#.replace(np.NaN, 0)

data = df.drop(["id",'diagnosis'], axis=1) #separate data from labels
lbl = df['diagnosis']

data = data.div(data.sum(axis=1), axis=0) #normalizate data to build the classifier

dataNoLabels = np.asmatrix(data) #data: dataframe to nArray

dataNoLabels, lbl = oversampling(dataNoLabels, lbl) #oversampling of minority labels

clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(dataNoLabels, lbl) #train RandomForest
predicoes = cross_val_predict(clf, dataNoLabels, lbl, cv=5)
print('Fase de Treino/Teste')
acc, f1s, precisao, recall = resultados(lbl, predicoes) #show results

#--------------------------------------------------------#

joblib.dump(clf, 'RF_up.joblib') #save the new model (limited) as joblib file
clf = joblib.load('RF_up.joblib')
usr_input = pd.read_csv('breast_cancer_usr.csv', sep=',', header=0)
data2 = usr_input.drop(["id",'diagnosis'], axis=1) #separate data from labels
lbl2 = usr_input['diagnosis']
simulacao = cross_val_predict(clf, data2, lbl2, cv=5)
print('Fase de execução com Usuário')
acc2, f1s2, precisao2, recall2 = resultados(lbl2, simulacao) #show results