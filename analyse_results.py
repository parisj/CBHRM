import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the provided CSV file
file_path = "results.csv"
data = pd.read_csv(file_path)
data = data.iloc[500:].reset_index(drop=True)
# Display the first few rows of the dataset to understand its structure
print(data.head())
# Calculate the mean and standard deviation of the Heart Rate
hr_mean = data["Heart Rate"].mean()
hr_std = data["Heart Rate"].std()
data["HR Change"] = data["Heart Rate"].diff().abs()
max_hr_change = data["HR Change"].max()
lines_with_greatest_change = data[data["HR Change"] == max_hr_change]

lines_with_greatest_change[
    ["Heart Rate", "Peaks", "Difference Between Peaks", "HR Change"]
]
# Find the index of the line with the greatest HR change and the index of the preceding line
index_of_max_change = lines_with_greatest_change.index[0]
index_of_preceding_line = index_of_max_change - 1

# Extracting the relevant lines from the dataset
line_with_max_change = data.loc[index_of_max_change]
preceding_line = data.loc[index_of_preceding_line]

print(
    line_with_max_change, preceding_line, index_of_max_change, index_of_preceding_line
)

# Identify significant deviations (e.g., more than 2 standard deviations from the mean)
fig, ax = plt.subplots()
ax.plot(data["HR Change"])
plt.show()
