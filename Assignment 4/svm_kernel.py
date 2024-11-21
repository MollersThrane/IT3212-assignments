from vehicle_preprocessing import preprocess_dataset
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load the vehicle dataset
data_dir = "./Datasets/Vehicle_detection"
X_train, X_test, y_train, y_test = preprocess_dataset(data_dir)

# Display the shape of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Initialize the SVM classifier
svm = SVC(kernel='rbf', gamma=0.001, C=1.0)

# Fit the classifier on the training data
svm.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = svm.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)
