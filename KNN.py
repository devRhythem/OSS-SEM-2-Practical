# Program NO.1
# Scikit learn library used to implementing Ml models
from sklearn.datasets import load_iris        # Dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data      # Features (flower measurements)
y = iris.target    # Target (flower species)

# 2. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# 3. Initialize KNN model
# n_neighbors = 3 → looks at 3 nearest neighbors to classify
knn = KNeighborsClassifier(n_neighbors=5)

# 4. Train the model
knn.fit(X_train, y_train)

# 5. Make predictions
y_pred = knn.predict(X_test)

# 6. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Sample predictions
print("First 5 Predictions:", y_pred[:5])
print("First 5 True Values:", y_test[:5])
print("End")
print("Khatam")
print("Tata")