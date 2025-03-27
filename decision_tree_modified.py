# Import libraries to work with CSV files and to graph the trees
import pandas as pd
from sklearn import tree
import graphviz
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

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
    #"Machine",
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

undersampler = RandomUnderSampler(random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)
 
print(Counter(y_res))

# Train the model and set the seed as 1410
clf = tree.DecisionTreeClassifier(random_state=1410)
clf = clf.fit(X, y)

# Print plot

dot_data = tree.export_graphviz(clf, out_file=None,
    feature_names=X.columns,
    class_names=["Malware", "Legitimate"],
    filled=True, rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('tek_tree_cleaned_variables', view=True).replace('\\', '/')


# Predict the data
predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
predict_data = predict_data[columns]
print(predict_data)
prediction = clf.predict(predict_data)

# Read the ID from the test CSV file
data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

#Creates the column for the predictions
data_2["Prediction"] = prediction
data_2.to_csv('Delta.csv', index=False)