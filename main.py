import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

happy_path = "happy_clean.csv"  # Replace with the actual file path
df_happy = pd.read_csv(happy_path)
sad_path = "sad_clean.csv"  # Replace with the actual file path
df_sad = pd.read_csv(sad_path)
neutral_path = "neutral2_data.csv"  # Replace with the actual file path
df_neutral = pd.read_csv(neutral_path)
# Calculate autocorrelation for each channel
autocorrelation_neutral = df_neutral.apply(lambda x: x.autocorr(), axis=0)
autocorrelation_happy = df_happy.apply(lambda x: x.autocorr(), axis=0)
autocorrelation_sad = df_sad.apply(lambda x: x.autocorr(), axis=0)

print(df_happy.head())
# Plot the autocorrelation data for each class
plt.figure(figsize=(12, 6))
plt.title("Autocorrelation for Different Emotion Classes")
plt.xlabel("Channel")
plt.ylabel("Autocorrelation Value")
plt.plot(autocorrelation_neutral, label="Neutral", marker='o')
plt.plot(autocorrelation_happy, label="Happy", marker='o')
plt.plot(autocorrelation_sad, label="Sad", marker='o')
plt.xticks(range(len(autocorrelation_neutral)), autocorrelation_neutral.index, rotation=90)
plt.legend()
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("autocorrelation_plot.png")

# Show the plot (optional)
plt.show()


# Calculate cross-correlation between channels
cross_correlation_neutral = df_neutral.corr()
cross_correlation_happy = df_happy.corr()
cross_correlation_sad = df_sad.corr()

# Plot the cross-correlation matrices for each class
plt.figure(figsize=(12, 6))
# Plot for Neutral
plt.subplot(131)
plt.title("Cross-Correlation Matrix - Neutral")
plt.imshow(cross_correlation_neutral, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

# Plot for Happy
plt.subplot(132)
plt.title("Cross-Correlation Matrix - Happy")
plt.imshow(cross_correlation_happy, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

# Plot for Sad
plt.subplot(133)
plt.title("Cross-Correlation Matrix - Sad")
plt.imshow(cross_correlation_sad, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.tight_layout()

# Save the plots as PNG files
plt.savefig("cross_correlation_plots.png")

# Show the plots (optional)
plt.show()
# Calculate other time-related statistics (e.g., mean, std, skewness, kurtosis)
mean_neutral = df_neutral.mean()
std_neutral = df_neutral.std()


mean_sad=df_sad.mean()
std_sad = df_sad.std()


mean_happy=df_happy.mean()
std_happy = df_happy.std()


data_neutral = pd.concat([mean_neutral, std_neutral], axis=1)
data_sad = pd.concat([mean_sad, std_sad], axis=1)
data_happy = pd.concat([mean_happy, std_happy], axis=1)

# Provide meaningful column names for each statistic in each DataFrame
data_neutral.columns = ['Mean', 'Std']
data_sad.columns = ['Mean', 'Std']
data_happy.columns = ['Mean', 'Std']

# Add an 'Emotion' column to each DataFrame
data_neutral['Emotion'] = 'Neutral'
data_sad['Emotion'] = 'Sad'
data_happy['Emotion'] = 'Happy'

# Concatenate the DataFrames for different emotions
data = pd.concat([data_neutral, data_sad, data_happy], axis=0)

sns.pairplot(data, hue='Emotion', diag_kind='kde')
plt.suptitle('Pair Plot of Descriptive Statistics by Emotion Class')
plt.savefig("statistayks2.png")
plt.show()
