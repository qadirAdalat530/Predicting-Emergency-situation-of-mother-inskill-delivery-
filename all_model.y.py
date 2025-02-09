# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:40:59 2023

@author: dell
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score


# Load the dataset
dataset = pd.read_csv('D:/projec final/My Project Dataset_Mother.csv')
dataset.head()

# Handle missing values
data = dataset.dropna()


# Split the dataset into input features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Scale the features using standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Separate the features and the target vaiable
X = dataset.drop('Mother alive', axis = 1)
y = dataset['Mother alive']

#Calculate the pearson correlation coefficient between each feature and the target variable
corr = X.corrwith(y)
#Rank the feartures based on their correlation values
ranked_feature = corr.abs().sort_values(ascending=False)
#Select the top-ranked features
num_feature = 8
selected_feature = ranked_feature[:num_feature]
print(selected_feature)

#Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
#Train the classifier
knn.fit(X_train, y_train)
# predict on test data
y_pred = knn.predict(X_test)
# Evaluate the performance of the classifier on the testing set
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy", accuracy)


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Create a Naive Bayes classifier
nb = GaussianNB()
# Train the calssifier
nb.fit(X_train, y_train)
#predict on test data
y_pred = nb.predict(X_test)
# Calculate accuracy for NB
accracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy : ", accuracy)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Create an SVM classifier
svm = SVC(kernel = 'linear')
# Train the classifier
svm.fit(X_train, y_train)
#predict on test data
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy", accuracy)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Create a neural network model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", accuracy)

precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Create a Random Forest classifier with 10000 trees
rf_classifier = RandomForestClassifier(n_estimators=10000, random_state=42)
# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy, precision, recall, and F1-score of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Random Forest Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Create a logistic Regression classifier
logreg = LogisticRegression()
# Train the classifier
logreg.fit(X_train, y_train)
# Predict on test data
y_pred = logreg.predict(X_test)

# Calculate the accuracy, precision, recall, and F1-score of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Logistic Regression :', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)


