import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Read customer_purchase_history dataset 
df=pd.read_csv("/content/Customer dataset.csv")
df.info()
df.shape
df.head(5)

x=df.iloc[:,[3,4]].values

# Find Within Cluster Sum of Square
WCSS=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
  kmeans.fit(x)
  WCSS.append(kmeans.inertia_)

plt.plot(range(1,11),WCSS)
plt.title("The ELBOW graph")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS Values")
plt.show()

kmeansmodel=KMeans(n_clusters=5,init="k-means++",random_state=0)
y_kmeans=kmeansmodel.fit_predict(x)

#Clustering Customer groups 
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=25,marker='*',c='cyan',label="customer 1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=25,marker='*',c='yellowgreen',label="customer 2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=25,marker='*',c='coral',label="customer 3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=25,marker='*',c='gold',label="customer 4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=25,marker='*',c='plum',label="customer 5")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=25,c='royalblue',marker='s',label='centroids')
plt.title("Clusters of Customers")
plt.xlabel("Annul Income in $")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
