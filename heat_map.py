#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Threshold variable for the correlation decision
threshold = 0.5

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

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = X.corr()

""""
#Correlation with output variable
cor_target = abs(cor["legitimate"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>threshold]
print(relevant_features)
"""
#Printing the map
sns.heatmap(cor, annot=False, cmap=plt.cm.Reds)
plt.show()
