#The Kaggle prediction was:
#   Private score: 0.49771
#   Public score: 0.50177

# Import library to work with CSV files
import pandas as pd

# Read the ID from the test CSV file
data = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Create a version with all the data to see the column names
data2 = pd.read_csv('malware-detection/test_data.csv')
print(data2.columns)

# Add the column with the predictions
data["Prediction"] = 0

# Write the CSV file to then submit
data.to_csv('AlphaZero.csv', index=False)

# Print the file to check that the amount of columns is correct
print(data)
