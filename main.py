# create streamlit app to load iris dataset from seaborn
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

df = sns.load_dataset('iris')
df
x = df.iloc[:, :-1]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

st.sidebar.title('Classifiers')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM'))
if classifier == 'KNN' :
  knn = KNeighborsClassifier(n_neighbors=10)
  knn.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  accuracy_score(y_test, y_pred)
if classifier == 'SVM' :
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  accuracy_score(y_test, y_pred)
if classifier == 'Decision Tree' :
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  accuracy_score(y_test, y_pred)

