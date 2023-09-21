import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the CSV files into DataFrames (replace 'file1.csv', 'file2.csv', 'file3.csv' with your file paths)

df_happy = pd.read_csv('happy_clean.csv',header=None)
df_sad = pd.read_csv('sad_clean.csv',header=None)

df_happy2 = pd.read_csv('happy_clean2.csv',header=None)
df_sad2 = pd.read_csv('sad_clean2.csv',header=None)
# Assuming the DataFrames have the same number of columns and are aligned by columns, concatenate them vertically.
# If the order is different or there are additional columns, you may need to adjust accordingly.
your_data = pd.concat([df_happy, df_sad,df_happy2,df_sad2], axis=0)

print(your_data)
# Create labels corresponding to the emotions (0 for neutral, 1 for happy, 2 for sad)
your_labels = np.concatenate([
    np.ones(len(df_happy)),
    np.full(len(df_sad), 2),
    np.ones(len(df_happy2)),
    np.full(len(df_sad2), 2)
])



# Define the sliding window size and step size
window_size = 18  # Number of measurements in each window
step_size = 72  # Number of measurements to move the window by


# Split your data into windows and corresponding labels
def create_sliding_windows(data, labels):
    windows = []
    window_labels = []

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        window_label = labels[i:i + window_size][0]  # Assuming all measurements in the window have the same label
        windows.append(window)
        window_labels.append(window_label)

    return windows, window_labels


# Apply the sliding window function to your data and labels
windows, window_labels = create_sliding_windows(your_data, your_labels)

print(windows)
# Split your windows and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(windows, window_labels, test_size=0.2, random_state=42)

# Flatten the windows (convert from 3D to 2D) if needed
X_train_flat = np.reshape(X_train, (len(X_train), -1))
X_test_flat = np.reshape(X_test, (len(X_test), -1))

# Train a machine learning model (e.g., Random Forest) on the flattened window data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_flat, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_flat)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
