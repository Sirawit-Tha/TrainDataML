from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

svm = SVR()
svm.fit(x.reshape(-1, 1), y)
y_pred = svm.predict(x.reshape(-1, 1))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.scatter(x, y)
plt.scatter(x, y_pred)

nn = MLPRegressor()
nn.fit(x.reshape(-1, 1), y)
y_pred = nn.predict(x.reshape(-1, 1))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.scatter(x, y)
plt.scatter(x, y_pred)

dt = DecisionTreeRegressor()
dt.fit(x.reshape(-1, 1), y)
y_pred = dt.predict(x.reshape(-1, 1))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.scatter(x, y)
plt.scatter(x, y_pred)

rf = RandomForestRegressor()
rf.fit(x.reshape(-1, 1), y)
y_pred = rf.predict(x.reshape(-1, 1))
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.scatter(x, y)
plt.scatter(x, y_pred)

st.header('😊 MACHINE LEARNING 😊')
st.write('By Mr.Sirawit Thajakan')

x = np.random.rand(100)
y = 2 * x + 1 + 0.2 * np.random.rand(100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

st.sidebar.title('Classifiers')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM','Decision Tree', 'Random Forest', 'Neural network' ))
k = st.sidebar.slider('Select K', 1, 20, 1)
if classifier == 'KNN' :
  knn = KNeighborsRegressor(n_neighbors=5)
  knn.fit(x.reshape(-1, 1), y)
  y_pred = knn.predict(x.reshape(-1, 1))
  plt.scatter(x, y)
  plt.plot(x, y_pred, color='red')
  plt.scatter(x, y)
  plt.scatter(x, y_pred)
