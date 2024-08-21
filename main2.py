from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
  st.pyplot(x, y)
  st.pyplot(x, y_pred, color='red')
  st.pyplot(x, y)
  st.pyplot(x, y_pred)
  
