import pandas as pd

# Load your existing files
fake_df = pd.read_csv("data/raw/Fake.csv")
true_df = pd.read_csv("data/raw/True.csv")

# Add labels
fake_df['label'] = 'FAKE'  # or 1 for numerical labels
true_df['label'] = 'REAL'  # or 0 for numerical labels

# Combine and save
combined_df = pd.concat([fake_df, true_df])
combined_df.to_csv("data/raw/fake_or_real_news.csv", index=False)

print("Combined dataset created successfully!")