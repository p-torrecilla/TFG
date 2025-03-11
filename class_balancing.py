# Import library to work with CSV files
import pandas as pd
import matplotlib.pyplot as plt


# Pull the data from the file
data = pd.read_csv('malware-detection/tek_data.csv')


# For each column, show the histogram
for i in data.columns:
    counts, bins, patches = plt.hist(data[i], color='#850501', edgecolor='black')
    
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2
        plt.text(x, height, int(count), ha='center', va='bottom', fontsize=10, color='black')


    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title(i)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
