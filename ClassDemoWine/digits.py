# import out DT classifier
from sklearn.datasets import load_digits
diGits_raw = load_digits()
# print(diGits.data.shape)
from sklearn import tree
# this is data numpy
import numpy as np

import matplotlib.pyplot as plt


# lets see our target names - these are the correct names each row of data is labeled as
print("\n\t *** Target Names ***")
for tn in diGits_raw.target_names:
    # rows
    print(tn)
# display feature names of diGits
print("\n\t *** Feature Names ***")
for fn in diGits_raw.feature_names:
    # columns
    print(fn)

# three samples of test data for diGits types 0, 1, 2, 3 -> 10
test_data_by_index = [10, 102, 200, 300, 400, 500, 600, 700, 1300, 1400, 1500, 1600, 1790]

# traning data - about 97%
trainging_target = np.delete(diGits_raw.target, test_data_by_index)
trainging_data = np.delete(diGits_raw.data, test_data_by_index, axis=0)

# testing data - removing 3 rows of test data
test_target = diGits_raw.target[test_data_by_index]
test_data = diGits_raw.data[test_data_by_index]

# create our DT
dt_classifier = tree.DecisionTreeClassifier()
# do the tranning
dt_classifier.fit(trainging_data, trainging_target)

# display results
print("\n\t*** Test Results ***")
print("Test Target actual lables are ", test_target)
print("Predictions by our DT labels are ", dt_classifier.predict(test_data))

plt.gray()
plt.matshow(diGits_raw.images[0])
plt.show()
