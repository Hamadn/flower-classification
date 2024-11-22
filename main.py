import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df["Species"] = iris.target

df["Species"] = df["Species"].map(
    {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
)

df.info()
df.head()
df.tail()


X = df.drop("Species", axis=1)
y = df["Species"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")

report = classification_report(y_test, y_pred)
print("Classification report: \n", report)
