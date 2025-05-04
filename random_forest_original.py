# Import libraries to work with CSV files and to graph the trees
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

start_time = time.time()


# Read from the tek CSV file
data = pd.read_csv('malware-detection/tek_data.csv')

# Define features and target
X = data.drop(columns=['ID', 'Machine', 'legitimate'])
y = data['legitimate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1410)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=1410)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}")

# Predict the data
predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
predict_data = predict_data[X.columns]
prediction = clf.predict(predict_data)

end_time = time.time()
total_duration = end_time - start_time
print(f"Time: {total_duration:.2f} seconds")

# Read the ID from the test CSV file
data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Creates the column for the predictions
data_2["Prediction"] = prediction
data_2.to_csv('RF_original.csv', index=False)
