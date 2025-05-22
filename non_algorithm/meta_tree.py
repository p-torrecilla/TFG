import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call
import matplotlib.pyplot as plt
import numpy as np

# Load the trained Random Forest model
with open('optuna_random_forest.sav', 'rb') as f:
    rf_model = pickle.load(f)

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

# Use the Random Forest to make predictions on the training data
rf_preds = rf_model.predict(X)

# Train a simple surrogate tree
meta_tree = DecisionTreeClassifier(max_depth=3, random_state=1410)
meta_tree.fit(X, rf_preds)


export_graphviz(
    meta_tree,
    out_file='meta_tree.dot',
    feature_names=columns,
    class_names=["Malware", "Legitimate"],
    filled=True,
    rounded=True,
    proportion=False,
    precision=0,
    impurity=False,
    label='none'
)

# Convert DOT to PNG
call(['dot', '-Tpng', 'meta_tree.dot', '-o', 'meta_tree.png', '-Gdpi=300'])

print("Clean meta tree saved as 'meta_tree.png'")