# Import libraries to work with CSV files and to graph the trees
import pandas as pd
from sklearn import tree
#import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import time

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

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=1410)


# Train the model and set the seed as 1410
clf = tree.DecisionTreeClassifier(random_state=1410)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.5f}")

# Print plot

#dot_data = tree.export_graphviz(clf, out_file=None,
#    feature_names=X.columns,
#    class_names=["Malware", "Legitimate"],
#    filled=True, rounded=True,
#    special_characters=True)
#graph = graphviz.Source(dot_data)
#graph.render('tek_tree_cleaned_variables', view=True).replace('\\', '/')


# Predict the data
predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
predict_data = predict_data[columns]
prediction = clf.predict(predict_data)

end_time = time.time()
total_duration = end_time - start_time
print(f"Time: {total_duration:.2f} seconds")

# Read the ID from the test CSV file
data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Creates the column for the predictions
data_2["Prediction"] = prediction
data_2.to_csv('AD_correlated.csv', index=False)