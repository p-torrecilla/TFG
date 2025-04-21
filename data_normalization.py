# Import library to work with CSV files
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import time

start_time = time.time()


# Pull the data from the file
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
print(X.describe())


# MinMaxing
x_copy_minmaxer = X.copy()
norm = MinMaxScaler().fit(x_copy_minmaxer)

x_norm = norm.transform(x_copy_minmaxer)
x_norm_data = pd.DataFrame(x_norm)
x_norm_data.columns = X.columns

print(x_norm_data.describe())

# Decision tree
def dec_tree():
    # Train the model and set the seed as 1410
    clf = tree.DecisionTreeClassifier(random_state=1410)
    clf = clf.fit(x_norm_data, y)

    # Predict the data
    predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
    predict_data = predict_data[columns]
    prediction = clf.predict(predict_data)

    end_time_dt = time.time()
    total_duration_dt = end_time_dt - start_time
    print(f"Time DT: {total_duration_dt:.2f} seconds")

    # Read the ID from the test CSV file
    data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

    # Creates the column for the predictions
    data_2["Prediction"] = prediction
    #data_2.to_csv('Zeta.csv', index=False)

def rand_forest():
    # Initialize and train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=1410)
    clf.fit(x_norm_data, y)

    # Predict the data
    predict_data = pd.read_csv('malware-detection/test_data.csv', header=0)
    predict_data = predict_data[columns]
    prediction = clf.predict(predict_data)

    end_time_rf = time.time()
    total_duration_rf = end_time_rf - start_time
    print(f"Time RF: {total_duration_rf:.2f} seconds")

    # Read the ID from the test CSV file
    data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

    # Creates the column for the predictions
    data_2["Prediction"] = prediction
    data_2.to_csv('Eta.csv', index=False)


dec_tree()
rand_forest()





























'''
# Z-scoring
x_copy_zscore = X.copy()
standard = StandardScaler().fit(x_copy_zscore)

x_stan = standard.transform(x_copy_zscore)
x_stan_data = pd.DataFrame(x_stan)
x_stan_data.columns = X.columns
'''