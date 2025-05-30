import optuna
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle 

start_time = time.time()

# Read from the tek CSV file
data = pd.read_csv('malware-detection/tek_data.csv')

# Define features and target
columns = [
    "AddressOfEntryPoint",
    "BaseOfData",
    "Characteristics",
    "CheckSum",
    "ExportNb",
    "FileAlignment",
    "ImageBase",
    "ImportsNbDLL",
    "ImportsNbOrdinal",
    "LoadConfigurationSize",
    "MajorImageVersion",
    "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion",
    "ResourcesMeanEntropy",
    "ResourcesMeanSize",
    "ResourcesMinEntropy",
    "SectionsMeanRawsize",
    "SectionsMinRawsize",
    "SectionsNb",
    "SizeOfCode",
    "SizeOfHeaders",
    "SizeOfHeapReserve",
    "SizeOfStackCommit",
    "SizeOfUninitializedData",
    "Subsystem"
]
X = data[columns]
y = data['legitimate']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1410)

end_reading_time = time.time()
reading_time = end_reading_time - start_time

# Objective function for Optuna
def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 20, 100)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight='balanced',
        random_state=1410
    )

    score = cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")
    return score.mean()

# Run Optuna optimization
start_optuna = time.time()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

end_optuna = time.time()

optuna_duration = end_optuna - start_optuna

# Print best parameters
print("Best parameters:", study.best_params)
print(f"Optimization time: {optuna_duration:.2f} seconds")


# Train final model with best parameters
start_forest = time.time()
best_clf = RandomForestClassifier(**study.best_params, random_state=1410)
best_clf.fit(X_train, y_train)

# save the model
filename = 'optuna_random_forest.sav'
pickle.dump(best_clf, open(filename, 'wb'))

# load the model
load_model = pickle.load(open(filename, 'rb'))

loaded_pred = load_model.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_pred)
print(f"Accuracy: {loaded_accuracy:.5f}")


y_pred = best_clf.predict(X_test)

# Predict the data
predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
predict_data = predict_data[columns]
prediction = best_clf.predict(predict_data)

end_forest_time = time.time()
total_duration = end_forest_time - start_forest + reading_time
print(f"Time: {total_duration:.2f} seconds")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}")

# Read the ID from the test CSV file
data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Creates the column for the predictions
data_2["Prediction"] = prediction
data_2.to_csv('RF_optuna.csv', index=False)
