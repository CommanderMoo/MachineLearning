# import libs
# importing out toy dataset
from sklearn.datasets import load_wine
# import out DT classifier
from sklearn import tree
# this is data numpy
import numpy as np
# special scikit command to load built in wine dataset
wine_raw = load_wine()

# lets see our target names - these are the correct names each row of data is labeled as
print("\n\t *** Target Names ***")
for tn in wine_raw.target_names:
    # rows
    print(tn)
# display feature names of wine
print("\n\t *** Feature Names ***")
for fn in wine_raw.feature_names:
    # columns
    print(fn)

# three samples of test data for wine types 0, 1, 2
test_data_by_index = [22, 102, 172]

# traning data - about 97%
trainging_target = np.delete(wine_raw.target, test_data_by_index)
trainging_data = np.delete(wine_raw.data, test_data_by_index, axis=0)

# testing data - removing 3 rows of test data
test_target = wine_raw.target[test_data_by_index]
test_data = wine_raw.data[test_data_by_index]

# create our DT
dt_classifier = tree.DecisionTreeClassifier()
# do the tranning
dt_classifier.fit(trainging_data, trainging_target)

# display results
print("\n\t*** Test Results ***")
print("Test Target actual lables are ", test_target)
print("Predictions by our DT labels are ", dt_classifier.predict(test_data))




