import pandas as pd

# 1. Load CSV files
df1 = pd.read_csv("trainingData.csv")
df2 = pd.read_csv("validationData.csv")

# 2. Check and report missing rows
missing_df1 = df1[df1.isna().any(axis=1)]
missing_df2 = df2[df2.isna().any(axis=1)]

print(f"trainingData.csv: {len(df1)} rows | {len(missing_df1)} rows with missing values")
print(f"validationData.csv: {len(df2)} rows | {len(missing_df2)} rows with missing values")

# 3. Concatenate datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Report missing rows in the combined dataset
missing_combined = combined_df[combined_df.isna().any(axis=1)]
print(f"Combined total: {len(combined_df)} rows | {len(missing_combined)} rows with missing values")

# 4. Drop categorical targets (FLOOR, BUILDINGID) and metadata columns
categorical_cols = [
    "FLOOR",
    "BUILDINGID",
    "SPACEID",
    "RELATIVEPOSITION",
    "USERID",
    "PHONEID",
    "TIMESTAMP",
]

cols_to_drop = [col for col in categorical_cols if col in combined_df.columns]
combined_df = combined_df.drop(columns=cols_to_drop)

# Drop any non-numeric (object/category) columns if present
combined_df = combined_df.select_dtypes(exclude=["object", "category"])

# 5. Rename continuous target columns with 'target_' prefix
continuous_targets = ["LONGITUDE", "LATITUDE"]

rename_dict = {
    col: f"target_{col.lower()}"
    for col in continuous_targets
    if col in combined_df.columns
}
combined_df = combined_df.rename(columns=rename_dict)

# 6. Save output file
combined_df.to_csv("ujiindoorloc_preprocessed.csv", index=False)
print(f"Saved to 'ujiindoorloc_preprocessed.csv' with shape {combined_df.shape}")