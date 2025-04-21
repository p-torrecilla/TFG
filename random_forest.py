# Import libraries to work with CSV files and to graph the trees
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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


# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=1410)
clf.fit(X, y)


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
#data_2.to_csv('Epsilon.csv', index=False)
