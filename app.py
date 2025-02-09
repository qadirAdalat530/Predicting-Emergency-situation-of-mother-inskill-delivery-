import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



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

# Create a Random Forest classifier with 1000 trees
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)

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

def predict_emergency_situation(mother_age, num_prev_pregnancies, num_miscarriage_stillbirth, diabetes, hypertension, cesarean_section, normal_delivery, gestational_age):
    input_data = np.array([[mother_age, num_prev_pregnancies, num_miscarriage_stillbirth, diabetes, hypertension, cesarean_section, normal_delivery, gestational_age]])
    prediction = rf_classifier.predict(input_data)[0]
    
    if prediction == 1:
        return 'Emergency situation'
    return 'No Emergency situation'

# Example usage
result = predict_emergency_situation(62, 3, 2, 1, 0, 0, 1, 32)
print(result)