import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess
titanic = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'sex', 'pclass', 'survived'])
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

X = titanic[['age', 'fare', 'sex', 'pclass']].values
Y = titanic['survived'].values

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
m, n = X.shape
weights = np.zeros(n)
bias = 0
learningRate = 0.01
iterations = 50000

# Gradient descent training
for i in range(iterations):
    linearModel = np.dot(X, weights) + bias
    yPred = sigmoid(linearModel)
    
    # Cost
    cost = (-1/m) * np.sum(Y*np.log(yPred + 1e-9) + (1-Y)*np.log(1 - yPred + 1e-9))
    
    # Gradients
    dw = (1/m) * np.dot(X.T, (yPred - Y))
    db = (1/m) * np.sum(yPred - Y)
    
    # Update
    weights -= learningRate * dw
    bias -= learningRate * db
    
    # Show progress
    if i % 200 == 0:
        print(f"Iteration {i:4d} | Cost: {cost:.4f}")

# Prediction function
def predict(X, weights, bias):
    linearModel = np.dot(X, weights) + bias
    yPred = sigmoid(linearModel)
    return [1 if i >= 0.5 else 0 for i in yPred]

# Evaluate
y_pred = predict(X, weights, bias)
accuracy = np.mean(y_pred == Y) * 100
print("Predictions:", y_pred[:10])
print(f"Accuracy: {accuracy:.2f}%")


#Visualize Decision Boundary (optional)
plt.figure(figsize=(7,5))
plt.scatter(X[:,0], X[:,1], c=Y, cmap='bwr', edgecolors='k')

x_values = np.array([min(X[:,0]), max(X[:,0])])
y_values = -(bias + weights[0]*x_values) / weights[1]

plt.plot(x_values, y_values, color='green',label='decision boundary')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.show()





# print(X,Y)
# print(size)
