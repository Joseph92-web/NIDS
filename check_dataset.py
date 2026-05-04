# check_dataset.py
import pandas as pd

print("=" * 50)
print("CHECKING KDDTest+.txt FORMAT")
print("=" * 50)

df = pd.read_csv('data/KDDTest+.txt', header=None)
print("First 5 rows:")
print(df.head())
print("\n" + "=" * 50)
print("First row (all columns):")
print(df.iloc[0, :])
print("\n" + "=" * 50)
print(f"Shape: {df.shape}")  # Should be (22544, 42) or (22544, 41)
print(f"Data types:\n{df.dtypes.value_counts()}")