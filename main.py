# Import library to work with CSV files
import pandas as pd
import random

# Read the ID from the test CSV file
data = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

#Creates the column for the predictions
data["Prediction"] = 0

# Adds the prediction to each ID
for i in range(1, len(data["Prediction"])):
    data.loc[i, "Prediction"] = random.randint(0,1)

# Write the CSV file to then submit
data.to_csv('Beta.csv', index=False)

# Print the file to check that the amount of columns is correct
print(data)
