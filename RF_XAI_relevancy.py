import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# === Load the model from disk ===
with open('optuna_random_forest.sav', 'rb') as f:
    model = pickle.load(f)

# === Define feature columns ===
columns = [
    "AddressOfEntryPoint",
    "BaseOfData",
    "Characteristics",
    "CheckSum",
    "ExportNb",
    "FileAlignment",
    "ImageBase",
    "ImportsNbDLL",
    "ImportsNbOrdinal",
    "LoadConfigurationSize",
    "MajorImageVersion",
    "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion",
    "ResourcesMeanEntropy",
    "ResourcesMeanSize",
    "ResourcesMinEntropy",
    "SectionsMeanRawsize",
    "SectionsMinRawsize",
    "SectionsNb",
    "SizeOfCode",
    "SizeOfHeaders",
    "SizeOfHeapReserve",
    "SizeOfStackCommit",
    "SizeOfUninitializedData",
    "Subsystem"
]

# === Load dataset for evaluation (optional step) ===
data = pd.read_csv('malware-detection/tek_data.csv')
X = data[columns]
y = data['legitimate']

# Optional: check model accuracy
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Model accuracy on tek_data: {accuracy:.4f}")

# === Load prediction dataset ===
predict_data = pd.read_csv('malware-detection/test_data.csv')
X_predict = predict_data[columns]
ids = predict_data["ID"]

# === Make predictions ===
predictions = model.predict(X_predict)

# === Save predictions with ID ===
results_df = pd.DataFrame({"ID": ids, "Prediction": predictions})
results_df.to_csv("RF_optuna.csv", index=False)

# === XAI with LIME ===

# Create the LIME explainer
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=columns,
    class_names=['Malware', 'Legitimate'],
    mode='classification',
    discretize_continuous=True
)

# Wrapped prediction to ensure column names are retained
def wrapped_predict_proba(x):
    return model.predict_proba(pd.DataFrame(x, columns=columns))

# === Aggregate LIME importances over multiple instances ===
num_instances = 100
feature_contributions = defaultdict(float)

# Extract base feature names (all valid feature names)
valid_feature_names = set(columns)

for i in range(min(num_instances, len(X_predict))):
    exp = explainer.explain_instance(
        data_row=X_predict.iloc[i].values,
        predict_fn=wrapped_predict_proba,
        num_features=len(columns)
    )

    for raw_feature, weight in exp.as_list():
        # Now we just map the raw_feature to a valid feature name from the list of columns
        # Extract the base name of the feature (e.g., "CheckSum" from "CheckSum <= 12345")
        feature_name = next((name for name in valid_feature_names if name in raw_feature), None)
        
        if feature_name:  # If a match is found, add the weight to the correct feature
            feature_contributions[feature_name] += abs(weight)

# === Normalize and sort features ===
sorted_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)

# Set number of top features to display
top_n = 10
top_features = sorted_features[:top_n]
features, importances = zip(*top_features)

# Normalize to percentages
importances = np.array(importances)
normalized_importances = 100 * importances / np.sum(importances)
cumulative_importance = np.cumsum(normalized_importances)

# === Plot bar chart + cumulative line ===
fig, ax1 = plt.subplots(figsize=(10, 12))

# Bar plot for individual importances
ax1.barh(features, normalized_importances, color='r', label='Feature Importance')
ax1.set_xlabel("Relative Importance (%)")
ax1.invert_yaxis()
ax1.grid(axis='x')

# ax1.tick_params(axis='y', labelsize=18)

# Line plot for cumulative ratio (on second y-axis)
ax2 = ax1.twiny()
ax2.plot(cumulative_importance, features, color='#636363', marker='o', label='Cumulative Importance')
ax2.set_xlim(0, 100)
ax2.set_xlabel("Cumulative Importance (%)")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='lower right')

plt.tight_layout()

plt.savefig("lime_feature_importance_top10.pdf", format="pdf")

plt.show()
