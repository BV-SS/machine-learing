 import pandas as pd
import numpy as np


from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns

#from sklearn.naive_bayes import GuassianNB

mdata=pd.read_csv(r'C:\Users\BHARTI VERMA\Documents\ML_project_datasec\data.csv')
print("imported")

mdata=pd.DataFrame(mdata)

print("The data have shape.............\n:",mdata.shape)
print("data have features as........... \n:",mdata.columns)
print(" data ..................\n :\n",mdata.head())
print("\n.........discription...........:\n",mdata.describe)

corr=mdata.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)



mdata.drop(['filename','tempo','beats','mfcc18','mfcc19'],1,inplace=True)
print("\ncolumns 'filename','tempo','beats','mfcc18', 'mfcc19' dropped")
print("new shape.....................\n",mdata.shape)


x=mdata.iloc[:,:24]
y=mdata.iloc[:,24:]

x.columns=[  'chroma_stft', 'rmse','spectral_centroid', 'spectral_bandwidth', 'rolloff',
            'zero_crossing_rate', 'mfcc1', 'mfcc2','mfcc3','mfcc4', 'mfcc5','mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
            'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17','mfcc20']

    
                       
y.columns=['label']


print(x)
print(y)
print(mdata.groupby('label').size())

#histogram
plt.hist(x,histtype='stepfilled')
plt.show()

#line graph 
mdata.plot(kind='line',title='line graph representation',figsize=(14,5))
plt.xlabel("labels")
plt.ylabel("max.value variation with time")
plt.show()

colors=range(mdata["mfcc4"].count())

  

#scatter plots
plt.scatter(x.chroma_stft,x.rmse,c=colors)
plt.xlabel("chroma_stft")
plt.ylabel("rmse")
plt.show()

plt.scatter(x.spectral_centroid,x.spectral_bandwidth,c=colors)
plt.xlabel("spectral_centroid")
plt.ylabel("spectral_bandwidth")
plt.show()

plt.scatter(x.rolloff,x.zero_crossing_rate,c=colors)
plt.xlabel("rolloff")
plt.ylabel("zero_crossing_rate")
plt.show()

plt.scatter(x.mfcc1,x.mfcc9,c=colors)
plt.xlabel("mfcc1")
plt.ylabel("mfcc9")
plt.show()

#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_test=sc.fit_transform(x_test)
X_train=sc.fit_transform(x_train)

''' model selection'''
#model=KNeighborsClassifier(n_neighbors=5,p=2,metric='euclidean')
model=svm.SVC()
#model=LogisticRegression(solver='lbfgs',max_iter=1000,multi_class='auto')
#model=GausianNB()
#model=DecisionTreeClassifier(random_state=0)
#model=RandomForestRegressor(n_estimators=20, random_state=0)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("y pred:\n", np.array(y_pred))

cm=confusion_matrix(y_test,y_pred)

print("\n confusion matrix :\n\n\n",cm)
accuracy=accuracy_score(y_test,y_pred)
print("\n accuracy score:",accuracy)

features=[[0.38],[0.24],[2116],[1952],[4196],[0.12],[-26.12],[107.3],[-46.81],[40.93],[-21.93],[24.81],[-18.94],[15.15],[-15.15],[12.26],[-15.123]]


for i in range(0,1):
    for j in range(0,7):
         inp=input(f'enter feature {i}')
          
         features[0].append(inp)
    
features=pd.DataFrame(features)
print(features)


def pred_out(features,model=model):
    genre=model.predict(features)
    print("this is",genre)

pred_out(features)
