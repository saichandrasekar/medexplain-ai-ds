# Databricks notebook source
# Load the CSV file
df = spark.read.csv(
    "/Volumes/workspace/medexplain-ai/medexplain/raw/kaggle-dataset-heart.csv",
    header=True,
    inferSchema=True
)

# Display the dataframe
display(df)

# COMMAND ----------

# Check nulls, data types, and value ranges per column
from pyspark.sql.functions import col, count, when, min, max, countDistinct

# Get column data types
print("=" * 80)
print("DATA TYPES")
print("=" * 80)
df.printSchema()

# Check nulls and basic statistics for each column
print("\n" + "=" * 80)
print("NULL COUNTS AND VALUE RANGES")
print("=" * 80)

# Get total row count
total_rows = df.count()
print(f"\nTotal rows: {total_rows}\n")

# Analyze each column
for column in df.columns:
    col_type = dict(df.dtypes)[column]
    null_count = df.filter(col(column).isNull()).count()
    null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
    distinct_count = df.select(countDistinct(column)).collect()[0][0]
    
    print(f"Column: {column}")
    print(f"  Type: {col_type}")
    print(f"  Nulls: {null_count} ({null_pct:.2f}%)")
    print(f"  Distinct values: {distinct_count}")
    
    # For numeric columns, show min/max
    if col_type in ['int', 'bigint', 'double', 'float', 'decimal', 'smallint', 'tinyint']:
        stats = df.select(min(column).alias('min'), max(column).alias('max')).collect()[0]
        print(f"  Range: [{stats['min']}, {stats['max']}]")
    # For string columns, show sample values if distinct count is reasonable
    elif col_type == 'string' and distinct_count <= 20:
        sample_values = df.select(column).distinct().limit(10).rdd.flatMap(lambda x: x).collect()
        print(f"  Sample values: {sample_values}")
    
    print()

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
display(df.describe())

# COMMAND ----------

from pyspark.sql.functions import mean, expr, col

# Separate numeric and categorical columns
numeric_cols = [field.name for field in df.schema.fields 
                if field.dataType.typeName() in ['integer', 'long', 'double', 'float', 'decimal']]
categorical_cols = [field.name for field in df.schema.fields 
                    if field.dataType.typeName() == 'string']

print("="*80)
print("CENTRAL TENDENCY MEASURES")
print("="*80)

# For numeric columns: calculate mean and median
if numeric_cols:
    print("\nNumeric Columns (Mean & Median):")
    print("-" * 80)
    
    # Calculate mean
    mean_values = df.select([mean(col(c)).alias(c) for c in numeric_cols]).collect()[0].asDict()
    
    # Calculate median using percentile
    median_exprs = [expr(f"percentile({c}, 0.5)").alias(c) for c in numeric_cols]
    median_values = df.select(median_exprs).collect()[0].asDict()
    
    for col_name in numeric_cols:
        print(f"\n{col_name}:")
        print(f"  Mean:   {mean_values[col_name]:.4f}")
        print(f"  Median: {median_values[col_name]:.4f}")

# For categorical columns: calculate mode
if categorical_cols:
    print("\n" + "="*80)
    print("Categorical Columns (Mode):")
    print("-" * 80)
    
    for col_name in categorical_cols:
        mode_row = df.groupBy(col_name).count().orderBy(col("count").desc()).first()
        if mode_row:
            print(f"\n{col_name}:")
            print(f"  Mode: {mode_row[0]} (count: {mode_row[1]})")

print("\n" + "="*80)

# COMMAND ----------

# DBTITLE 1,Cell 4
# All columns are already numeric (integers), no encoding needed
df_encoded = df

# Display the result
display(df_encoded)

# COMMAND ----------

# DBTITLE 1,Cell 5
# Check if target column exists and verify it's binary
from pyspark.sql.functions import col, countDistinct

# Identify the target column (common names for heart disease prediction)
possible_target_cols = ['target', 'heart_disease', 'output', 'diagnosis', 'label']
target_col = None

for col_name in possible_target_cols:
    if col_name in df_encoded.columns:
        target_col = col_name
        break

if target_col is None:
    # If not found, check last column (often the target)
    target_col = df_encoded.columns[-1]
    print(f"No standard target column found. Using last column: {target_col}")
else:
    print(f"Target column identified: {target_col}")

# Check if target is binary
unique_values = df_encoded.select(countDistinct(col(target_col))).collect()[0][0]
actual_values = sorted([row[0] for row in df_encoded.select(target_col).distinct().collect()])

print(f"\nTarget column '{target_col}' analysis:")
print(f"  Number of unique values: {unique_values}")
print(f"  Actual values: {actual_values}")

if unique_values == 2 and set(actual_values) == {0, 1}:
    print(f"  ✓ Target is binary (0/1) - keeping as-is")
    print(f"\n{'='*80}")
    print("CLASSIFICATION THRESHOLDS DOCUMENTATION")
    print("='*80}")
    print(f"\nTarget Variable: {target_col}")
    print(f"  - Class 0: No heart disease (negative class)")
    print(f"  - Class 1: Heart disease present (positive class)")
    print(f"\nDefault Decision Threshold: 0.5")
    print(f"  - Predictions ≥ 0.5 → Class 1 (positive)")
    print(f"  - Predictions < 0.5 → Class 0 (negative)")
    print(f"\nThreshold Selection Guidelines:")
    print(f"  - Balanced accuracy: Use 0.5")
    print(f"  - Minimize false negatives (medical screening): Lower threshold (e.g., 0.3-0.4)")
    print(f"  - Minimize false positives (reduce unnecessary tests): Higher threshold (e.g., 0.6-0.7)")
    print(f"  - Optimize for specific metric: Use ROC/precision-recall curve analysis")
    print(f"\nRecommendation: For heart disease prediction, a lower threshold (0.3-0.4) is often")
    print(f"preferred to minimize false negatives, as missing a case (Type II error) has")
    print(f"more serious consequences than a false alarm (Type I error).")
    print("="*80)
else:
    print(f"  ⚠ Warning: Target is not strictly binary 0/1")
    print(f"  Consider transforming to binary format if needed for classification")

# Display class distribution
print(f"\nClass Distribution:")
df_encoded.groupBy(target_col).count().orderBy(target_col).show()

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from pyspark.sql.functions import col
import numpy as np
import pandas as pd

# Identify continuous (numeric) columns
# Exclude the target column if present
target_col = 'target'  # Adjust if your target column has a different name
numeric_cols = [field.name for field in df_encoded.schema.fields 
                if field.dataType.typeName() in ['integer', 'long', 'double', 'float', 'decimal']
                and field.name != target_col]

print(f"Continuous columns to scale: {numeric_cols}")

# Convert to pandas for StandardScaler
df_pandas = df_encoded.toPandas()

# Separate features and target
X = df_pandas[numeric_cols]
y = df_pandas[target_col] if target_col in df_pandas.columns else None

# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create scaled dataframe
df_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

# Add back the target column if it exists
if y is not None:
    df_scaled[target_col] = y.values

# Display scaled data
print(f"\nScaled features shape: {df_scaled.shape}")
print(f"\nScaling parameters:")
for i, col_name in enumerate(numeric_cols):
    print(f"  {col_name}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}")

display(df_scaled.head(10))

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split into 80% train, 20% test with random seed for reproducibility
X = df_scaled.drop(columns=['target'])
y = df_scaled['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution in both sets
)

print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTraining set class distribution:")
print(y_train.value_counts().sort_index())
print(f"\nTest set class distribution:")
print(y_test.value_counts().sort_index())

# COMMAND ----------

# Write the final processed data to CSV
output_path = "/Volumes/workspace/medexplain-ai/medexplain/processed/heart_disease_processed.csv"

# Convert to pandas and save as CSV
df_scaled.to_csv(output_path, index=False)

print(f"Successfully saved processed data to: {output_path}")
print(f"Shape: {df_scaled.shape}")
print(f"Columns: {list(df_scaled.columns)}")