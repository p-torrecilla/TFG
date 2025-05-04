# Import library to work with CSV files
import pandas as pd
import random
import time

start_time = time.time()

# Read the ID from the test CSV file
data = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

#Creates the column for the predictions
data["Prediction"] = 0

# Adds the prediction to each ID
for i in range(1, len(data["Prediction"])):
    data.loc[i, "Prediction"] = random.randint(0,1)


end_time = time.time()
total_duration = end_time - start_time
print(f"Time: {total_duration:.2f} seconds")


# Write the CSV file to then submit
data.to_csv('Random.csv', index=False)