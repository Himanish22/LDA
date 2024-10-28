import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

wine_data = pd.read_csv('wine.csv')
features = wine_data.drop(columns='Target')
target = wine_data['Target']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train_lda, y_train)

lda_predictions = lda.predict(X_test_scaled)
lda_accuracy = accuracy_score(y_test, lda_predictions)
lda_precision = precision_score(y_test, lda_predictions, average='weighted')
lda_recall = recall_score(y_test, lda_predictions, average='weighted')
lda_confusion_matrix = confusion_matrix(y_test, lda_predictions)

print("LDA Model Evaluation:")
print(f"Accuracy: {lda_accuracy:.2f}")
print(f"Precision: {lda_precision:.2f}")
print(f"Recall: {lda_recall:.2f}")
print("Confusion Matrix:")
print(lda_confusion_matrix)

logistic_predictions = logistic_model.predict(X_test_lda)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
logistic_precision = precision_score(y_test, logistic_predictions, average='weighted')
logistic_recall = recall_score(y_test, logistic_predictions, average='weighted')
logistic_confusion_matrix = confusion_matrix(y_test, logistic_predictions)

print("\nLogistic Regression Model Evaluation:")
print(f"Accuracy: {logistic_accuracy:.2f}")
print(f"Precision: {logistic_precision:.2f}")
print(f"Recall: {logistic_recall:.2f}")
print("Confusion Matrix:")
print(logistic_confusion_matrix)

x_min, x_max = X_train_lda[:, 0].min() - 1, X_train_lda[:, 0].max() + 1
y_min, y_max = X_train_lda[:, 1].min() - 1, X_train_lda[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = logistic_model.predict(grid_points)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, edgecolor='k', marker='o')
plt.title('Logistic Regression Decision Boundaries in LDA 2D Space')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.show()
