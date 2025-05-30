# The Kaggle prediction was:
#   Private score: 0.75777
#   Public score: 0.75542

# Import libraries to work with CSV files and to graph the trees
import pandas as pd
from sklearn import tree
import graphviz

# Read from the tek CSV file
data = pd.read_csv('malware-detection/tek_data.csv')

# Define features and target
X = data.drop(columns=['ID', 'Machine', 'legitimate'])
y = data['legitimate']

# Train the model and set the seed as 1410
clf = tree.DecisionTreeClassifier(random_state=1410)
clf = clf.fit(X, y)

# Print plot
dot_data = tree.export_graphviz(clf, out_file=None,
    feature_names=X.columns,
    class_names=["Malware", "Legitimate"],
    filled=True, rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('tek_tree', view=True).replace('\\', '/')

# Predict the data
predict_data = pd.read_csv('malware-detection/test_data.csv', usecols=range(1,55), header=0)
predict_data = predict_data.drop(columns=["ID"], errors='ignore')
predict_data = predict_data.drop(columns=["Machine"])
prediction = clf.predict(predict_data)

# Read the ID from the test CSV file
data_2 = pd.read_csv('malware-detection/test_data.csv', usecols=["ID"])

#Creates the column for the predictions
data_2["Prediction"] = prediction
data_2.to_csv('Gamma.csv', index=False)
