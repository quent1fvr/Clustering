import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as scp
import sklearn.metrics  as metrics 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA




hotels= pd.read_csv("hotels.csv")
#print(hotels.head())

df= hotels.drop(['NOM', 'PAYS', 'ETOILE',],axis="columns")
#print(df.head())

df_NOM=hotels.iloc[:,0]
df_ETOILE=hotels.iloc[:,2]

#print(df_NOM)
#print(df_ETOILE)

df_COLUMNS=df.columns
#print(df_COLUMNS)

#print(df.corr())

#pd.plotting.scatter_matrix(df)
#plt.show()

#variables les plus corrélées positivement : prix et la cuisine / cuisine et confort / plage et sport 
# variables les moins corrélées : confort et plage / prix et chambre / sport et confort 
# variables les plus corrélées négativement : prix et chalmbre 

df2= df.to_numpy()
scaler = StandardScaler()
Z= scaler.fit_transform(df2)

#print(Z.var())
#print(Z.mean())

#classification ascendante hiérarchique

CAh_ward = scp.linkage(Z, method='ward', metric='euclidean',optimal_ordering="True")
CAh_centroid = scp.linkage(Z, method='complete', metric='euclidean',optimal_ordering="True")

#print(CAh_single)

t=6
Dendro_CAh_single= scp.dendrogram(CAh_ward, color_threshold=t)
plt.show()

Dendro_CAh_centroid= scp.dendrogram(CAh_centroid, color_threshold=0)

plt.show()

#on choisit 4 ou 5 clusters 



#PARTIE KMEANS 

K_means= KMeans(5,n_init=1,init="random")
K_means1= KMeans(5,n_init=1,init="random")

data= K_means.fit_transform(Z)
data1= K_means1.fit_transform(Z)

#print(K_means.labels_)
#print(K_means.inertia_)

ARI=metrics.adjusted_rand_score(K_means1.labels_,K_means.labels_)
#print(ARI)

K_means2= KMeans(5,n_init=10,init="random")
K_means21= KMeans(5,n_init=10,init="random")
data2= K_means2.fit_transform(Z)
data21= K_means21.fit_transform(Z)
ARI2=metrics.adjusted_rand_score(K_means2.labels_,K_means21.labels_)
#print(ARI2)


K_means3= KMeans(5,n_init=10,init="k-means++")
K_means31= KMeans(5,n_init=10,init="k-means++")
data3= K_means3.fit_transform(Z)
data31= K_means31.fit_transform(Z)
ARI3=metrics.adjusted_rand_score(K_means3.labels_,K_means31.labels_)
#print(ARI3)



#utilisation de la métrique "silhouette"
#faire varier le nombre de clusters de 2 à 10
res = np.arange(9,dtype="double")
for k in np.arange(9):
    km = KMeans(n_clusters=k+2, n_init=10,init="k-means++")
    km.fit(Z)
    res[k] = metrics.silhouette_score(Z,km.labels_)
    #print(res)
    
#graphique
import matplotlib.pyplot as plt
#plt.title("Silhouette")
#plt.xlabel("# de clusters")
#plt.plot(np.arange(2,11,1),res)
#plt.show()

#on choisit la valeur de K égale à 2

K_means_final= KMeans(4,n_init=10,init="random")
data_final= K_means_final.fit_transform(Z)



idk = np.argsort(K_means_final.labels_)
#affichage des observations et leurs groupes

print(K_means_final.labels_)
print(idk)

idk = np.argsort(K_means_final.labels_)
print(df_NOM[idk],K_means_final.labels_[idk])
#print(pd.DataFrame(K_means_final.labels_[idk],df_NOM[idk],df_ETOILE[idk]))

#print(pd.hotels(hotels.index[idk],K_means_final.labels_[idk]))
#distances aux centres de classes des observations

acp = PCA(2,svd_solver='full')
coord = acp.fit_transform(data_final)


for couleur,k in zip(['red','blue','lawngreen','aqua'],[0,1,2,3,4]):
    
    plt.scatter(coord[K_means_final.labels_==k,0],coord[K_means_final.labels_==k,1],c=couleur)
plt.title("2 premiers axes principaux")

for i,label in enumerate(df_ETOILE):
    plt.annotate(label,(coord[i,0],coord[i,1]))
plt.show()
plt.show()

