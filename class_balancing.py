# Import library to work with CSV files
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler


# Pull the data from the file
data = pd.read_csv('malware-detection/tek_data.csv')


# For each column, show the histogram
for i in data.columns:
    '''
    counts, bins, patches = plt.hist(data[i], color='#850501', edgecolor='black')
    
    for count, patch in zip(counts, patches):
        height = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2
        # Writing a label with the exact amount on each column
        plt.text(x, height, int(count), ha='center', va='bottom', fontsize=10, color='black')


    plt.xlabel('Valores')
    plt.ylabel('Frecuencia')
    plt.title(i)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
    '''
    # Crating the pie chart for the 'legitimate' column distribution
    if i == 'legitimate':
        value_counts = data[i].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(value_counts, colors=["r", "#636363"], labels=value_counts.index, explode=[0, 0.1], shadow=True, autopct='%1.1f%%')
        plt.title(i)
        plt.show()


        # Distribution of 'legitimate' after applying the undersampler
        X = data
        y = data['legitimate']
        undersampler = RandomUnderSampler(random_state=1410)
        X_res, y_res = undersampler.fit_resample(X, y)
        value_counts = X_res[i].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(value_counts, colors=["r", "#636363"], labels=value_counts.index, explode=[0, 0.1], shadow=True, autopct='%1.1f%%')
        plt.title(i)
        plt.show()
