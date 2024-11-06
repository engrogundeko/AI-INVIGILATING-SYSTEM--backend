from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example confusion matrix values
TP = 45  # True Positives
FN = 0  # False Negatives
FP = 16  # False Positives
TN = 0 # True Negatives

# Calculate y_true and y_pred based on the confusion matrix
y_true = [1] * TP + [1] * FN + [0] * FP + [0] * TN
y_pred = [1] * TP + [0] * FN + [1] * FP + [0] * TN

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(accuracy, precision, recall, f1)