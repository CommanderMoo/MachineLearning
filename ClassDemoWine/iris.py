from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
# load dataset
iris_raw = load_iris()

# display names
print("\n\t *** Target Names ***")
for tn in iris_raw.target_names:
    # rows
    print(tn)

# display feature names of iris
print("\n\t *** Feature Names ***")
for fn in iris_raw.feature_names:
    # columns
    print(fn)

# iris_raw samples
hardCode_test = [49, 99, 149]

# traning
training_target = np.delete(iris_raw.target, hardCode_test)
training_data = np.delete(iris_raw.data, hardCode_test, axis=0)

# testing data - removing 3 rows of test data
test_target = iris_raw.target[hardCode_test]
test_data = iris_raw.data[hardCode_test]

# create our DT
dt_classifier = tree.DecisionTreeClassifier()
# do the tranning
dt_classifier.fit(training_data, training_target)

# display results
print("\n\t*** Test Results ***")
print("Test Target actual labels are ", test_target)
print("Predictions by our DT labels are ", dt_classifier.predict(test_data))

# working with help from hinton
# problem being
# hardCode_test = [[#,#,#]] <- double brackets
# (four dimensional array) cant occur
