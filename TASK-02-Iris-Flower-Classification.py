import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset('iris')
iris['species'], categories = pd.factorize(iris['species'])
print(iris.describe())
# print(iris.head())
print(iris.isna().sum())

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(iris.petal_length,iris.petal_width,iris.species)
ax.set_xlabel('Petal_Length_Cm')
ax.set_ylabel('Petal_Width_Cm')
ax.set_zlabel('species')
plt.title('3d Scatter Plot For Petal')
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(iris.sepal_length,iris.sepal_width,iris.species)
ax.set_xlabel('Sepal_Length_Cm')
ax.set_ylabel('Sepal_Width_Cm')
ax.set_zlabel('species')
plt.title('3d Scatter Plot For Sepal')
plt.show()


sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species").set_title("2d Scatter Plot For Sepal")
plt.show()
sns.scatterplot(data=iris,x="petal_length",y="petal_width",hue="species").set_title("2d Scatter Plot For Petal")
plt.show()

# Using Elbow Tcehnique

k_rng = range(1, 10)
sse = []

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(iris[['petal_length', 'petal_width']])
    sse.append(km.inertia_)

plt.xlabel('k_rng')
plt.ylabel("Sum of Squared errors")
plt.plot(k_rng, sse)
plt.plot()

# Applying KMean Algorithm

km = KMeans(n_clusters=3,random_state=0,)
y_predicted = km.fit_predict(iris[['petal_length','petal_width']])
print(y_predicted)
iris['cluster']=y_predicted
print(iris.head(150))

# Measuring Accuracy

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(iris.species, iris.cluster)
print(cm)

true_labels = iris.species
predicted_labels= iris.cluster

cm = confusion_matrix(true_labels, predicted_labels)
class_labels = ['Setosa', 'versicolor', 'virginica']

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Fill matrix with values
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='white')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
