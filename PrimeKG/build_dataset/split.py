# split_triples.py

import pandas as pd
import random

# Set random seed for reproducibility
random.seed(42)

# Load the CSV file
input_file = '../data/BindData/triplet_full.csv'
df = pd.read_csv(input_file)

# Shuffle the data
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split indices
total = len(df_shuffled)
train_end = int(total * 0.8)
valid_end = train_end + int(total * 0.1)
# Split the data
train_df = df_shuffled.iloc[:train_end]
valid_df = df_shuffled.iloc[train_end:valid_end]
test_df = df_shuffled.iloc[valid_end:]

# Save to CSV files
train_df.to_csv('../data/BindData/train.csv', index=False)
valid_df.to_csv('../data/BindData/valid.csv', index=False)
test_df.to_csv('../data/BindData/test.csv', index=False)

print("Data has been split and saved as train.csv, valid.csv, and test.csv")
