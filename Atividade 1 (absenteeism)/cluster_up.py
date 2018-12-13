import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.externals import joblib

df = pd.read_csv('Absenteeism_at_work.csv', sep=';', header=0).replace(np.NaN, 0)
data = df.div(df.sum(axis=1), axis=0) #normalizate data to build the classifier
data = np.asmatrix(data) #data: dataframe to nArray

fig, ax = plt.subplots()
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(data)
    print(i, kmeans.inertia_)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS')  # within cluster sum of squares

ax.set(xlabel='Clusters', ylabel='Inercia',
       title='Elbow method: INERCIA')
ax.grid()
fig.savefig("elbow-wcss.png")
plt.show()
plt.show()

# Cluster data using k-means to predict classes
model = KMeans(n_clusters=3, random_state=1).fit(data)
# Get predicted classes
labels = model.labels_
print('Silhueta: ', silhouette_score(data, labels))

joblib.dump(model, 'Kmeans_up.joblib') #save the new model (limited) as joblib file
#------------------------------------------------------------#

usr_input = pd.read_csv('usr_input2.csv', sep=';', header=0).replace(np.NaN, 0)
clf = joblib.load('Kmeans_up.joblib')
print('Com o modelo salvo: ', clf.predict(usr_input))
labels2 = clf.labels_
print('Silhueta: ', silhouette_score(usr_input, labels2))