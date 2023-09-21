import pandas as pd
import numpy as np
# Read the input CSV file into a DataFrame
input_file = "happy_data.csv"
df = pd.read_csv(input_file,header=None)

# Create a list to store the calculated values
calculated_values = []

# Define the values for A153
A153 = [181.7241333, 166.7707333, 467.0684, 175.8852667, 327.6746, 481.8638667, 58.7264, 83.4146, 1078.745667,
        72.90813333, 533.2928667, 33.97466667, 165.9942667, 119.6166, 357.404, 1115.693133, 676.0401333, 306.3088667]

# Create a new DataFrame to store the calculated values
calculated_df = pd.DataFrame()

# Calculate and store the values for each row in the input DataFrame
for index, row in df.iterrows():
    calculated_row = []
    for i in range(len(row)):
        A_x = row[i]
        if i < len(A153):
            A_153 = A153[i]
            calculated_value = abs(np.log10(A_153 / A_x)) if A_x != 0 else ""
            calculated_row.append(calculated_value)
        else:
            calculated_row.append("")
    calculated_df = calculated_df.append(pd.Series(calculated_row), ignore_index=True)

# Save the new DataFrame to a new CSV file
output_file = "happy_clean.csv"
calculated_df.to_csv(output_file, index=False)

print("New CSV file saved as 'sad_clean.csv'")