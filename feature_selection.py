# Import library to work with CSV files and to use combinations
import pandas as pd
from itertools import combinations

# Read from the tek CSV file
data = pd.read_csv("malware-detection/tek_data.csv")

# Create the variable for the results to go into
results = []

# For every pair of columns (with no repetition), save their correlation
for col1, col2 in combinations(data.columns, 2):
    correlation = data[col1].corr(data[col2])
    results.append({"Column A": col1, "Column B": col2, "Correlation": correlation})

# Save the results as a dataframe
correlation_data = pd.DataFrame(results)

# Save the resulting dataframe in a .csv file
correlation_data.to_csv("correlation_results.csv", index=False)