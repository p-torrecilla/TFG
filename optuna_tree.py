import time
import optuna
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import graphviz
from imblearn.under_sampling import RandomUnderSampler

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

undersampler = RandomUnderSampler(random_state=1410)
X_res, y_res = undersampler.fit_resample(X, y)

end_reading_time = time.time()
reading_time = end_reading_time - start_time

# Objective function for Optuna
def objective(trial):
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=1410
    )

    score = cross_val_score(clf, X_res, y_res, cv=5, scoring="accuracy")
    return score.mean()

# Timer: Optuna optimization
start_optuna = time.time()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

end_optuna = time.time()
optuna_duration = end_optuna - start_optuna

print("Best parameters:", study.best_params)
print(f"Optimization time: {optuna_duration:.2f} seconds")

# Timer
start_tree = time.time()

best_clf = DecisionTreeClassifier(**study.best_params, random_state=1410)
best_clf.fit(X_res, y_res)


# Predict the data
predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
predict_data = predict_data[columns]
prediction = best_clf.predict(predict_data)


end_tree_time = time.time()
total_duration = end_tree_time - start_tree + reading_time
print(f"Time: {total_duration:.2f} seconds")

# Read the ID from the test CSV file
data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Creates the column for the predictions
data_2["Prediction"] = prediction
data_2.to_csv('Theta.csv', index=False)


# Visualize the tree
dot_data = export_graphviz(
    best_clf,
    out_file=None,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="png", cleanup=True)
