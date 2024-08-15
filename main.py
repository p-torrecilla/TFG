# Import library to work with CSV files
import pandas as pd

# Read the ID from the test CSV file
data = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

# Add the column with the (fake) predictions
data["Prediction"] = 0

# Write the CSV file to then submit
data.to_csv('Alpha.csv', index=False)

# Print the file to check that the amount of columns is correct
print(data)
