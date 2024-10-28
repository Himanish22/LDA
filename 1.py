import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris_df = pd.read_csv('Iris.csv')
features = iris_df.iloc[:, :-1].values
labels = iris_df.iloc[:, -1].values

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

lda_model = LinearDiscriminantAnalysis(n_components=2)
lda_features = lda_model.fit_transform(scaled_features, encoded_labels)

pca_model = PCA(n_components=2)
pca_features = pca_model.fit_transform(scaled_features)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for idx, label in zip(range(len(label_encoder.classes_)), label_encoder.classes_):
    plt.scatter(lda_features[encoded_labels == idx, 0], lda_features[encoded_labels == idx, 1], label=label)
plt.title('LDA: Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()

plt.subplot(1, 2, 2)
for idx, label in zip(range(len(label_encoder.classes_)), label_encoder.classes_):
    plt.scatter(pca_features[encoded_labels == idx, 0], pca_features[encoded_labels == idx, 1], label=label)
plt.title('PCA: Iris Dataset')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()

plt.tight_layout()
plt.show()
