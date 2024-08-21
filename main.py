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
from sklearn.neighbors import KNeighborsRegressor












st.header('😊 MACHINE LEARNING 😊')
st.write('By Mr.Sirawit Thajakan')

df = sns.load_dataset('iris')
df
x = df.iloc[:, :-1]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

st.sidebar.title('Classifiers')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM','Decision Tree', 'Random Forest', 'Neural network' ))
k = st.sidebar.slider('Select K', 1, 20, 1)
if classifier == 'KNN' :
  knn = KNeighborsClassifier(n_neighbors=10)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'SVM' :
  svm = SVC()
  svm.fit(x_train, y_train)
  y_pred = svm.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'Decision Tree' :
  dt = DecisionTreeClassifier()
  dt.fit(x_train, y_train)
  y_pred = dt.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'Random Forest' :
  rf = RandomForestClassifier()
  rf.fit(x_train, y_train)
  y_pred = rf.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)
if classifier == 'Neural network' :
  nn = MLPClassifier()
  nn.fit(x_train, y_train)
  y_pred = nn.predict(x_test)
  acc = accuracy_score(y_test, y_pred)
  st.write(acc)


st.bar_chart(df['species'].value_counts())

a = np.random.rand(100)
b = 2 * x + 1 + 0.2 * np.random.rand(100)
plt.scatter(a, b)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(a.reshape(-1, 1), b)
b_pred = knn.predict(a.reshape(-1, 1))
plt.scatter(a, b)
plt.plot(a, b_pred, color='red')
plt.scatter(a, b)
plt.scatter(a, b_pred)

