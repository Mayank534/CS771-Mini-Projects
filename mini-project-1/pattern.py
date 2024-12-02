import csv
import numpy as np

# Load data from the CSV file
filename = "trial.csv"

# Function to extract features from input string
def extract_features(input_str):
    digits = [int(d) for d in input_str]
    sum_digits = sum(digits)
    distinct_digits = len(set(digits))
    num_zeros = digits.count(0)
    even_count = sum(1 for d in digits if d % 2 == 0)
    odd_count = sum(1 for d in digits if d % 2 != 0)
    low_range = sum(1 for d in digits if 0 <= d <= 3)
    mid_range = sum(1 for d in digits if 4 <= d <= 6)
    high_range = sum(1 for d in digits if 7 <= d <= 9)
    
    return {
        "sum_digits": sum_digits,
        "distinct_digits": distinct_digits,
        "num_zeros": num_zeros,
        "even_count": even_count,
        "odd_count": odd_count,
        "low_range": low_range,
        "mid_range": mid_range,
        "high_range": high_range
    }

# Read the CSV file and split data by label
label_0_features = []
label_1_features = []

with open(filename, newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for row in csvreader:
        input_str = row[0]
        label = int(row[1])
        features = extract_features(input_str)
        
        if label == 0:
            label_0_features.append(features)
        else:
            label_1_features.append(features)

# Calculate statistics for each label
def calculate_statistics(feature_list):
    keys = feature_list[0].keys()
    stats = {}
    
    for key in keys:
        values = [f[key] for f in feature_list]
        stats[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values)
        }
        
    return stats

# Calculate statistics for label 0
label_0_stats = calculate_statistics(label_0_features)

# Calculate statistics for label 1
label_1_stats = calculate_statistics(label_1_features)

# Output the statistics for each label group
print("Statistics for Label 0:")
for feature, stat in label_0_stats.items():
    print(f"{feature}: mean={stat['mean']}, std={stat['std']}, min={stat['min']}, max={stat['max']}")

print("\nStatistics for Label 1:")
for feature, stat in label_1_stats.items():
    print(f"{feature}: mean={stat['mean']}, std={stat['std']}, min={stat['min']}, max={stat['max']}")
