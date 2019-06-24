# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:13:51 2019

@author: Alexia
"""


from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import KMeans
import numpy as np
from sklearn import cluster
import networkx as nx
from collections import defaultdict
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation


#------------------------------------------------------------------------------
#-------------------------------  INSERT THE DATA ---------------------------------------------
#------------------------------------------------------------------------------ 


mikos = int(input('''-- Insert the number of clusters you want to do your analysis with : '''))
k_clusters = mikos
file = input('''-- Insert the name of your file  {ex:new.xlsx} :''')

#------------------------------------------------------------------------------
#-------------------------------  OPEN DATA ---------------------------------------------
#------------------------------------------------------------------------------ 

#-open the dataset that you want to run your analysis on 
d=pd.read_excel(file)
# - keep the names here of all the genes ......sos the column must be called 'gene '
names=d['gene']
# - keep the number of genes in your dataset 
m=len(names)-1
print('\n-- Your genes are beeing processed ...')
# take out the names 
pi=d.drop(['gene'], axis=1)
#- keep  all the values in a variable called TI
ΤΙ= pi.iloc[:,0:].values
name=np.array(names)
X = np.array(ΤΙ)

#-------------------------------------------------------------------------------
#----------------------------DO THE FITTING ----------------------------------
#------------------------------------------------------------------------------

kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(X)
agglom = AgglomerativeClustering(n_clusters=k_clusters, linkage="ward").fit(X)

m=kmeans.labels_
n= agglom.labels_

#-------------------------------------------------------------------------------
#----------------------------SAVE THE RESULTS  ----------------------------------
#------------------------------------------------------------------------------


teliko=pd.DataFrame(list(zip(name,m)),columns=['GENES','CLUSTER '])
teliko.to_excel('kMEANS_cluster.xlsx', engine='xlsxwriter')
teliko=pd.DataFrame(list(zip(name,n)),columns=['GENES','CLUSTER '])
teliko.to_excel('agglom_cluster.xlsx', engine='xlsxwriter')

print('\n-- You are done, Please check your results with the following names :')
print('                > first file : kMEANS_cluster.xlsx') 
print('                > second file : agglom_cluster.xlsx')



#t=kmeans.predict([[0, 0], [12, 3]])
#print(t)