# Import library to work with CSV files
import pandas as pd


# Read from the tek CSV file
data = pd.read_csv('malware-detection/tek_data.csv')


# Creates a dataframe with checks if the values in each cell are null
null_data = data.notnull()


# Creates lists for he columns with null data and unique values
null_columns = []
unique_columns = []


# Checks every column to see if they have null values
for i in data.columns:
    if not null_data[i].any():
        null_columns.append(i)

    temp_set = set(data[i])
    if len(temp_set) == len(data["ID"]):
        unique_columns.append(i)


print("Columns with null data:")
print(*null_columns)
print("Columns with all unique values:")
print(*unique_columns)


# ARMADILLO
print("Finished")
