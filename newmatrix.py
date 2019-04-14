# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:24:40 2019

@author: Alexia
"""



import numpy as np
import re 
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm, grid_search
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



#---------------------------------------------------------------------------
#--------------------------FUNCTIONS----------------------------------------
#---------------------------------------------------------------------------
#-this function will clear your dataset in case there are some nan values or empty cells 

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


#---------------------------------------------------------------------------
#-------------------------OPEN DATA ----------------------------------------
#--------------------------------------------------------------------------    



df=pd.read_excel('good1.xlsx')
df=clean_dataset(df)
#-keep in x the variable and as y the validation 
x = df.iloc[:,1:].values
y = df.iloc[:,0].values
#-rename them 
data_input = x
data_output = y
#-set parameters for the kfold validation
kf = KFold(10, n_folds = 5, shuffle=True)
#-set parameters for the classifiers    
rf_class = RandomForestClassifier(n_estimators=10)
log_class = LogisticRegression()
svm_class = svm.SVC()
nn_class = KNeighborsClassifier(n_neighbors=3)
svc_class= SVC(kernel="linear", C=0.025)
gausian_class= GaussianProcessClassifier(1.0 * RBF(1.0))
dtc_class = DecisionTreeClassifier(max_depth=5)
mpl_class = MLPClassifier(alpha=1)
abc_class = AdaBoostClassifier()
bnb_class= GaussianNB()


accu=[]#-- here we will keep all the accuracies of each classifier 

print("Random Forests: ")
print(cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracy1 = cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracy1)
print("Accuracy of Random Forests is: " , accuracy1)

print("\n\nsvm-linear: ")
print(cross_val_score(svc_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracysvc = cross_val_score(svc_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracysvc)
print("Accuracy of svm-linear is: " , accuracysvc)

print("\n\nGaussian process classifier: ")
print(cross_val_score(gausian_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracygausian = cross_val_score(gausian_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracygausian)
print("Accuracy of  Gaussian process classifier is: " , accuracygausian)

print("\n\nDesicion tree classifier : ")
print(cross_val_score(dtc_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracydtc = cross_val_score(dtc_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracydtc)
print("Accuracy of Desicion tree classifier is: " , accuracydtc)

print("\n\nMPL: ")
print(cross_val_score(mpl_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracympl = cross_val_score(mpl_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracympl)
print("Accuracy of MPL Classifier is: " , accuracympl)


print("\n\nAdaBoostClassifier: ")
print(cross_val_score(abc_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracyabc = cross_val_score(abc_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracyabc)
print("Accuracy of AdaBoostClassifier is: " , accuracyabc)


print("\n\nGaussianNB:")# default is rbf
print(cross_val_score(bnb_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracybnb = cross_val_score(bnb_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracybnb)
print("Accuracy of GaussianNB is: " , accuracybnb)

print("\n\nSVM:")# default is rbf
print(cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracy2 = cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracy2)
print("Accuracy of SVM is: " , accuracy2)


print("\n\nLog:")
print(cross_val_score(log_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracy3 = cross_val_score(log_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracy3)
print("Accuracy of nLog is: " , accuracy3)

print("\n\nNN:")
print(cross_val_score(nn_class, data_input, data_output, scoring='accuracy', cv = 10))
accuracy = cross_val_score(nn_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
accu.append(accuracy)
print("Accuracy of NN is: " , accuracy)


#### -- here we will display the user which classifier we suggest them to use for their analysis 

bac=max(accu)
name =[]

if bac == accuracy1:
    name.append('Random Forest')
elif bac == accuracy2:
    name.append('SVM')
elif bac == accuracy3:
    name.append('nLog')
elif bac == accuracysvc:
    name.append('SVC-LINEAR')
elif bac == accuracygausian:
    name.append('Gausian process')
elif bac == accuracydtc:
    name.append('Decision Tree')
elif bac == accuracympl:
    name.append('MPL')
elif bac == accuracyabc:
    name.append('AdaBoost')
elif bac == accuracybnb:
    name.append('GaussianNB')
else:
    name.append('knn')

#-------------------------------------------------------------------------------------------
#-display all the results 
print("\n\n--Summing up :")

classi=['Random Forests','SVM','nLog','SVC-linear','Gaussianprocess','DecisionTree','MPL','AdaBoost','GaussianNB','NN']
accur=[accuracy1,accuracy2,accuracy3,accuracysvc,accuracygausian,accuracydtc,accuracympl,accuracyabc,accuracybnb,accuracy]
tog=zip(classi,accur)
teliko=pd.DataFrame(list(zip(classi,accur)),columns=['Classifier','Accuracy'])
tel=teliko.sort_values(by=['Accuracy'], ascending=False)
tel.to_excel('accuracies.xlsx', engine='xlsxwriter')
print(tel)
print('\nThe best accuracy was ', bac,'and was achieved with the ',name[0],'classifier')

print("\n-------------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------------")

classifiers = int(input('''-- Insert with which classifier you want to forceed the analysis:  
 _________________________________
|                                |
| 1) knn :               1       |
| 2) Randomforest :      2       |
| 3) LogisticRegretion:  3       | 
| 4) SVM :               4       |
| 5) SVM-linear:         5       |
| 6) GaussianProcess:    6       |
| 7) DecisionTree :      7       |
| 8) MPL :               8       |
| 9) ΑdaBoost:           9       |
| 10 GaussianNB:        10       |
| 11) all :              0       | 
|________________________________|  -- type   classifier {ex:4} : '''))

#---------------------------------------------------------------------------
#--------------------------FUNCTIONS----------------------------------------
#---------------------------------------------------------------------------

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def stringator(listind):
    stri=" ".join(str(x) for x in listind)
    T=re.sub("\s+", ",", stri.strip())
    return T


#---------------------------------------------------------------------------
#-------------------------OPEN TRAINING DATA ----------------------------------------
#--------------------------------------------------------------------------    

df=pd.read_excel('good1.xlsx')
#df=clean_dataset(df)
#-keep in x the variable and as y the validation 
x = df.iloc[:,1:].values
y = df.iloc[:,0].values
#-rename the variables 
data_input = x
data_output = y

#---------------------------------------------------------------------------
#-------------------------OPEN REAL DATA ----------------------------------------
#--------------------------------------------------------------------------  
#-open the dataset that you want to run your analysis on 
d=pd.read_excel('new.xlsx')
#d=clean_dataset(df)
# - keep the names here of all the genes ......sos the column must be called 'gene '
names=d['gene']
# - format the matrix so than there will only be int inside 
m=len(names)-1
print('\n-- There are in total',m,'genes in your dataset')
pi=d.drop(['gene'], axis=1)
#- keep  all the values in a variable called X 
X = pi.iloc[:,0:].values
tixera=[]# here are saved in good form all the values for predict 
#mikos=1 # here i define how many i want to run the test on 
mikos = int(input('''-- Insert number of genes you want to test : '''))
for t in range(mikos):
    M=X[t+1]
    MM=(int(x) for x in M)
    l=stringator(MM)
    tixera.append(l)
   
#------------------------------------------------------------------------
#--------------------------LEARNNG PART ---------------------------------
#-------------------------------------------------------------------------

#-Do the fitting on our data .
knn = KNeighborsClassifier(n_neighbors=3)
rf = RandomForestClassifier(n_estimators=10)
lg= LogisticRegression()
svm = svm.SVC()
svc= SVC(kernel="linear", C=0.025)
gausian= GaussianProcessClassifier(1.0 * RBF(1.0))
dtc= DecisionTreeClassifier(max_depth=5)
mpl= MLPClassifier(alpha=1)
abc= AdaBoostClassifier()
bnb= GaussianNB()

svm.fit(x,y)
knn.fit(x,y)
rf.fit(x,y)
lg.fit(x,y)
svc.fit(x,y)
gausian.fit(x,y)
dtc.fit(x,y)
mpl.fit(x,y)
abc.fit(x,y)
bnb.fit(x,y)




#------------------------------------------------------------------------
#--------------------------PREDICTION PART KNN---------------------------------
#-------------------------------------------------------------------------

if classifiers == 1 or classifiers == 0:
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=knn.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_knn.xlsx', engine='xlsxwriter')
    
    if  classifiers == 2 or classifiers == 0:
        
        #------------------------------------------------------------------------
        #--------------------------PREDICTION PART RF---------------------------------
        #-------------------------------------------------------------------------
        
        pred=[]
        for st in range(mikos):
            t=tixera[st]
            ti=[int(s) for s in t.split(',')]
            ni=rf.predict([ti])
            pred.append(ni[0])
        #------------------------------------------------------------------------
        #--------------------------Create final matrix ---------------------------------
        #-------------------------------------------------------------------------
        # given that in training 1 was as good target and as a bad one 
        tava=[]
        for pr in pred :
            if pr > 0.95 :
                t=' GOOD potential target'
                tava.append(t)
            else :
                t=' BAD potential target'
                tava.append(t)
        name=[]
        for st in range(mikos):
            t=names[st]
            name.append(t)
            
        
            
        #for st in range(mikos):
           # print('The gene',name[st],'is according to our algorithm',tava[st])
           
        teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
        teliko.to_excel('output_rf.xlsx', engine='xlsxwriter')
        
        if  classifiers == 3 or classifiers == 0:
            
            #------------------------------------------------------------------------
            #--------------------------PREDICTION PART RF---------------------------------
            #-------------------------------------------------------------------------
            
            pred=[]
            for st in range(mikos):
                t=tixera[st]
                ti=[int(s) for s in t.split(',')]
                ni=lg.predict([ti])
                pred.append(ni[0])
            #------------------------------------------------------------------------
            #--------------------------Create final matrix ---------------------------------
            #-------------------------------------------------------------------------
            # given that in training 1 was as good target and as a bad one 
            tava=[]
            for pr in pred :
                if pr > 0.95 :
                    t=' GOOD potential target'
                    tava.append(t)
                else :
                    t=' BAD potential target'
                    tava.append(t)
            name=[]
            for st in range(mikos):
                t=names[st]
                name.append(t)
                
            
                
            #for st in range(mikos):
               # print('The gene',name[st],'is according to our algorithm',tava[st])
               
            teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
            teliko.to_excel('output_lg.xlsx', engine='xlsxwriter')
            
            if  classifiers == 4 or classifiers == 0 :
                
                #------------------------------------------------------------------------
                #--------------------------PREDICTION PART RF---------------------------------
                #-------------------------------------------------------------------------
                
                pred=[]
                for st in range(mikos):
                    t=tixera[st]
                    ti=[int(s) for s in t.split(',')]
                    ni=svm.predict([ti])
                    pred.append(ni[0])
                #------------------------------------------------------------------------
                #--------------------------Create final matrix ---------------------------------
                #-------------------------------------------------------------------------
                # given that in training 1 was as good target and as a bad one 
                tava=[]
                for pr in pred :
                    if pr > 0.95 :
                        t=' GOOD potential target'
                        tava.append(t)
                    else :
                        t=' BAD potential target'
                        tava.append(t)
                name=[]
                for st in range(mikos):
                    t=names[st]
                    name.append(t)
                    
                
                    
                #for st in range(mikos):
                   # print('The gene',name[st],'is according to our algorithm',tava[st])
                   
                teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                teliko.to_excel('output_svm.xlsx', engine='xlsxwriter')
                if classifiers == 5 or classifiers == 0:
    
                    #------------------------------------------------------------------------
                    #--------------------------PREDICTION PART RF---------------------------------
                    #-------------------------------------------------------------------------
                    
                    pred=[]
                    for st in range(mikos):
                        t=tixera[st]
                        ti=[int(s) for s in t.split(',')]
                        ni=svc.predict([ti])
                        pred.append(ni[0])
                    #------------------------------------------------------------------------
                    #--------------------------Create final matrix ---------------------------------
                    #-------------------------------------------------------------------------
                    # given that in training 1 was as good target and as a bad one 
                    tava=[]
                    for pr in pred :
                        if pr > 0.95 :
                            t=' GOOD potential target'
                            tava.append(t)
                        else :
                            t=' BAD potential target'
                            tava.append(t)
                    name=[]
                    for st in range(mikos):
                        t=names[st]
                        name.append(t)
                        
                    
                        
                    #for st in range(mikos):
                       # print('The gene',name[st],'is according to our algorithm',tava[st])
                       
                    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                    teliko.to_excel('output_svc.xlsx', engine='xlsxwriter')
                    if classifiers == 6  or classifiers == 0:
    
                        #------------------------------------------------------------------------
                        #--------------------------PREDICTION PART RF---------------------------------
                        #-------------------------------------------------------------------------
                        
                        pred=[]
                        for st in range(mikos):
                            t=tixera[st]
                            ti=[int(s) for s in t.split(',')]
                            ni=gausian.predict([ti])
                            pred.append(ni[0])
                        #------------------------------------------------------------------------
                        #--------------------------Create final matrix ---------------------------------
                        #-------------------------------------------------------------------------
                        # given that in training 1 was as good target and as a bad one 
                        tava=[]
                        for pr in pred :
                            if pr > 0.95 :
                                t=' GOOD potential target'
                                tava.append(t)
                            else :
                                t=' BAD potential target'
                                tava.append(t)
                        name=[]
                        for st in range(mikos):
                            t=names[st]
                            name.append(t)
                            
                        
                            
                        #for st in range(mikos):
                           # print('The gene',name[st],'is according to our algorithm',tava[st])
                           
                        teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                        teliko.to_excel('output_gausianprocess.xlsx', engine='xlsxwriter')
                        if classifiers == 7 or classifiers == 0 :
    
                            #------------------------------------------------------------------------
                            #--------------------------PREDICTION PART RF---------------------------------
                            #-------------------------------------------------------------------------
                            
                            pred=[]
                            for st in range(mikos):
                                t=tixera[st]
                                ti=[int(s) for s in t.split(',')]
                                ni=dtc.predict([ti])
                                pred.append(ni[0])
                            #------------------------------------------------------------------------
                            #--------------------------Create final matrix ---------------------------------
                            #-------------------------------------------------------------------------
                            # given that in training 1 was as good target and as a bad one 
                            tava=[]
                            for pr in pred :
                                if pr > 0.95 :
                                    t=' GOOD potential target'
                                    tava.append(t)
                                else :
                                    t=' BAD potential target'
                                    tava.append(t)
                            name=[]
                            for st in range(mikos):
                                t=names[st]
                                name.append(t)
                                
                            
                                
                            #for st in range(mikos):
                               # print('The gene',name[st],'is according to our algorithm',tava[st])
                               
                            teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                            teliko.to_excel('output_dtc.xlsx', engine='xlsxwriter')
                            if classifiers == 8  or classifiers == 0 :
    
                                #------------------------------------------------------------------------
                                #--------------------------PREDICTION PART RF---------------------------------
                                #-------------------------------------------------------------------------
                                
                                pred=[]
                                for st in range(mikos):
                                    t=tixera[st]
                                    ti=[int(s) for s in t.split(',')]
                                    ni=mpl.predict([ti])
                                    pred.append(ni[0])
                                #------------------------------------------------------------------------
                                #--------------------------Create final matrix ---------------------------------
                                #-------------------------------------------------------------------------
                                # given that in training 1 was as good target and as a bad one 
                                tava=[]
                                for pr in pred :
                                    if pr > 0.95 :
                                        t=' GOOD potential target'
                                        tava.append(t)
                                    else :
                                        t=' BAD potential target'
                                        tava.append(t)
                                name=[]
                                for st in range(mikos):
                                    t=names[st]
                                    name.append(t)
                                    
                                
                                    
                                #for st in range(mikos):
                                   # print('The gene',name[st],'is according to our algorithm',tava[st])
                                   
                                teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                                teliko.to_excel('output_mpl.xlsx', engine='xlsxwriter')
                                if classifiers == 9 or classifiers == 0 :
                                    
                                    #------------------------------------------------------------------------
                                    #--------------------------PREDICTION PART RF---------------------------------
                                    #-------------------------------------------------------------------------
                                    
                                    pred=[]
                                    for st in range(mikos):
                                        t=tixera[st]
                                        ti=[int(s) for s in t.split(',')]
                                        ni=abc.predict([ti])
                                        pred.append(ni[0])
                                    #------------------------------------------------------------------------
                                    #--------------------------Create final matrix ---------------------------------
                                    #-------------------------------------------------------------------------
                                    # given that in training 1 was as good target and as a bad one 
                                    tava=[]
                                    for pr in pred :
                                        if pr > 0.95 :
                                            t=' GOOD potential target'
                                            tava.append(t)
                                        else :
                                            t=' BAD potential target'
                                            tava.append(t)
                                    name=[]
                                    for st in range(mikos):
                                        t=names[st]
                                        name.append(t)
                                        
                                    
                                        
                                    #for st in range(mikos):
                                       # print('The gene',name[st],'is according to our algorithm',tava[st])
                                       
                                    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                                    teliko.to_excel('output_adaboost.xlsx', engine='xlsxwriter')
                                    if classifiers == 10 or classifiers == 0 :
    
                                        #------------------------------------------------------------------------
                                        #--------------------------PREDICTION PART RF---------------------------------
                                        #-------------------------------------------------------------------------
                                        
                                        pred=[]
                                        for st in range(mikos):
                                            t=tixera[st]
                                            ti=[int(s) for s in t.split(',')]
                                            ni=bnb.predict([ti])
                                            pred.append(ni[0])
                                        #------------------------------------------------------------------------
                                        #--------------------------Create final matrix ---------------------------------
                                        #-------------------------------------------------------------------------
                                        # given that in training 1 was as good target and as a bad one 
                                        tava=[]
                                        for pr in pred :
                                            if pr > 0.95 :
                                                t=' GOOD potential target'
                                                tava.append(t)
                                            else :
                                                t=' BAD potential target'
                                                tava.append(t)
                                        name=[]
                                        for st in range(mikos):
                                            t=names[st]
                                            name.append(t)
                                            
                                        
                                            
                                        #for st in range(mikos):
                                           # print('The gene',name[st],'is according to our algorithm',tava[st])
                                           
                                        teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
                                        teliko.to_excel('output_bnb.xlsx', engine='xlsxwriter')
                                
                                    
                        
                    
elif  classifiers == 2 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=rf.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_rf.xlsx', engine='xlsxwriter')
    
elif  classifiers == 3 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=lg.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_lg.xlsx', engine='xlsxwriter')
    
elif classifiers == 4 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=svm.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_svm.xlsx', engine='xlsxwriter')
    
elif classifiers == 5 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=svc.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_svc.xlsx', engine='xlsxwriter')
    
elif classifiers == 6 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=gausian.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_gausianprocess.xlsx', engine='xlsxwriter')
    
elif classifiers == 7 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=dtc.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_dtc.xlsx', engine='xlsxwriter')
    
elif classifiers == 8 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=mpl.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_mpl.xlsx', engine='xlsxwriter')

elif classifiers == 9 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=abc.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_adaboost.xlsx', engine='xlsxwriter')

elif classifiers == 10 :
    
    #------------------------------------------------------------------------
    #--------------------------PREDICTION PART RF---------------------------------
    #-------------------------------------------------------------------------
    
    pred=[]
    for st in range(mikos):
        t=tixera[st]
        ti=[int(s) for s in t.split(',')]
        ni=bnb.predict([ti])
        pred.append(ni[0])
    #------------------------------------------------------------------------
    #--------------------------Create final matrix ---------------------------------
    #-------------------------------------------------------------------------
    # given that in training 1 was as good target and as a bad one 
    tava=[]
    for pr in pred :
        if pr > 0.95 :
            t=' GOOD potential target'
            tava.append(t)
        else :
            t=' BAD potential target'
            tava.append(t)
    name=[]
    for st in range(mikos):
        t=names[st]
        name.append(t)
        
    
        
    #for st in range(mikos):
       # print('The gene',name[st],'is according to our algorithm',tava[st])
       
    teliko=pd.DataFrame(list(zip(name,tava)),columns=['GENES','TARGET VALIDATION '])
    teliko.to_excel('output_bnb.xlsx', engine='xlsxwriter')

if classifiers == 0 :
    print('\n-- Since you have chosen to use all classifiers, a cross validation of all results will be done.')
    genes=[]
    counter=0
    
    outputs=['output_adaboost.xlsx','output_rf.xlsx','output_gausianprocess.xlsx','output_svm.xlsx','output_lg.xlsx','output_knn.xlsx','output_dtc.xlsx','output_svc.xlsx','output_mpl.xlsx','output_bnb.xlsx']
    for files in outputs :
        counter+=1
        df=pd.read_excel(files)
        d=np.array(df)
        for x in d:
            if x[1] == ' GOOD potential target':
                genes.append(x[0])
        print('done processing the :',counter,'file')
    
    counted_genes=Counter(genes)
    name=[]
    number=[]
    a1_sorted_keys = sorted(counted_genes, key=counted_genes.get, reverse=True)
    for r in a1_sorted_keys:
        name.append(r)
        number.append(counted_genes[r])
    
    df=pd.DataFrame(list(zip(name,number)),columns=['GENES','NO OCCURRENCES '])
    #df=pd.DataFrame.from_dict(counted_genes,orient='index',columns=['number of times validated as a good potential target'])
    d=np.array(df)
    dd=[]
    for x in d:
        if x[1]>1:
            dd.append(x)
    l=pd.DataFrame(dd,columns=['GENES','NO OCCURRENCES '])
    print(l)
    df.to_excel('crossvalidated.xlsx', engine='xlsxwriter')
