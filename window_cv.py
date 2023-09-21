import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the CSV files into DataFrames (replace 'file1.csv', 'file2.csv', 'file3.csv' with your file paths)
df_neutral = pd.read_csv('neutral_data.csv',header=None)
df_happy = pd.read_csv('happy_data.csv',header=None)
df_sad = pd.read_csv('sad_data.csv',header=None)

df_neutral2 = pd.read_csv('neutral2_data.csv',header=None)
df_happy2 = pd.read_csv('happy2_data.csv',header=None)
df_sad2 = pd.read_csv('sad2_data.csv',header=None)
# Assuming the DataFrames have the same number of columns and are aligned by columns, concatenate them vertically.
# If the order is different or there are additional columns, you may need to adjust accordingly.
your_data = pd.concat([df_neutral, df_happy, df_sad,df_neutral2,df_happy2,df_sad2], axis=0)

print(your_data)
# Create labels corresponding to the emotions (0 for neutral, 1 for happy, 2 for sad)
your_labels = np.concatenate([
    np.zeros(len(df_neutral)),
    np.ones(len(df_happy)),
    np.full(len(df_sad), 2),
    np.zeros(len(df_neutral2)),
    np.ones(len(df_happy2)),
    np.full(len(df_sad2), 2)
])



# Define the number of folds for cross-validation
n_splits = 5

# Initialize a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
accuracies = cross_val_score(clf, your_data, your_labels, cv=kf, scoring='accuracy')

# Print cross-validation results
print("Cross-Validation Results:")
for fold, accuracy in enumerate(accuracies, start=1):
    print(f"Fold {fold}: Accuracy = {accuracy:.2f}")

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy across {n_splits}-Fold Cross-Validation: {mean_accuracy:.2f}")



