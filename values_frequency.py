import pandas as pd


# Read from the tek CSV file
data = pd.read_csv('malware-detection/tek_data.csv')


excel = "values_frequency.xlsx"


# Create an Excel writer object
with pd.ExcelWriter(excel) as writer:
    # Iterate over each column to calculate value frequencies
    for column in data.columns:
        value_counts = data[column].value_counts(normalize=True) * 100
        freq_df = pd.DataFrame({"Value": value_counts.index, "Frequency": [freq for freq in value_counts.values], "Count": data[column].value_counts()})
        freq_df.to_excel(writer, sheet_name=column, index=False)
    

# ARMADILLO
print("Finished")