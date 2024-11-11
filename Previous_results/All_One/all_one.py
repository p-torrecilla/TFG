#The Kaggle prediction was:
#   Private score: 0.50228
#   Public score: 0.49822

# Import library to work with CSV files
import pandas as pd

# Read the ID from the test CSV file
data = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Create a version with all the data to see the column names
data2 = pd.read_csv('malware-detection/test_data.csv')
print(data2.columns)

# Add the column with the predictions
data["Prediction"] = 1

# Write the CSV file to then submit
data.to_csv('AlphaOne.csv', index=False)

# Print the file to check that the amount of columns is correct
print(data)
