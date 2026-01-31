"""
Data Preprocessing Pipeline for Employee Burnout Dataset
Prepares the dataset for machine learning algorithms by:
- Handling missing values
- Encoding categorical variables
- Removing low-correlation features
- Feature scaling
- Train-test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Features to remove based on analysis
FEATURES_TO_REMOVE = [
    'Mental Fatigue Score',      # Too close to Burn Rate (correlation: 0.94)
    'Designation',               # Bad feature as specified
    'Team Size',                 # Very low correlation (0.002)
    'Years in Company',          # Very low correlation (0.001)
    'Employee ID',               # Identifier, not a feature
    'Date of Joining'            # Temporal data, already have Years in Company
]

# Target variable
TARGET = 'Burn Rate'

# Random state for reproducibility
RANDOM_STATE = 42

# Test size for train-test split
TEST_SIZE = 0.2

# Create output directory for processed data
os.makedirs('processed_data', exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("DATA PREPROCESSING PIPELINE FOR EMPLOYEE BURNOUT ANALYSIS")
print("="*80)

print("\n1. Loading Dataset...")
df = pd.read_csv('Dataset/enriched_employee_dataset(1).csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Store original shape for comparison
original_shape = df.shape
print(f"   Original features: {df.columns.tolist()}")

# =============================================================================
# INITIAL DATA EXPLORATION
# =============================================================================

print("\n2. Initial Data Overview...")
print(f"   Target variable: {TARGET}")
print(f"   Missing values before processing:")
missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]
if len(missing_before) > 0:
    for col, count in missing_before.items():
        print(f"      - {col}: {count} ({count/len(df)*100:.2f}%)")
else:
    print("      - No missing values")

# =============================================================================
# FEATURE CORRELATION ANALYSIS
# =============================================================================

print("\n3. Analyzing Feature Correlations with Target Variable...")

# Calculate correlations only for numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if TARGET in numerical_cols:
    correlations = df[numerical_cols].corr()[TARGET].sort_values(ascending=False)
    print(f"\n   Feature Correlations with {TARGET}:")
    for feature, corr in correlations.items():
        if feature != TARGET:
            print(f"      {feature:.<50} {corr:>7.4f}")

# =============================================================================
# REMOVE SPECIFIED FEATURES
# =============================================================================

print("\n4. Removing Specified Features...")
features_removed = []
for feature in FEATURES_TO_REMOVE:
    if feature in df.columns:
        df = df.drop(columns=[feature])
        features_removed.append(feature)
        print(f"   ✓ Removed: {feature}")

print(f"\n   Total features removed: {len(features_removed)}")
print(f"   Remaining features: {df.shape[1]} (including target)")

# =============================================================================
# HANDLE MISSING VALUES
# =============================================================================

print("\n5. Handling Missing Values...")

# First, remove rows where target variable is missing
if df[TARGET].isnull().sum() > 0:
    rows_before = len(df)
    df = df.dropna(subset=[TARGET])
    rows_removed = rows_before - len(df)
    print(f"   ✓ Removed {rows_removed} rows with missing target variable")

# For remaining features, handle missing values
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove(TARGET)  # Exclude target from imputation

categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Impute numerical features with median
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"   ✓ Imputed {col} with median: {median_value:.2f}")

# Impute categorical features with mode
for col in categorical_features:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"   ✓ Imputed {col} with mode: {mode_value}")

print(f"\n   Remaining rows after handling missing values: {len(df)}")

# =============================================================================
# ENCODE CATEGORICAL VARIABLES
# =============================================================================

print("\n6. Encoding Categorical Variables...")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(categorical_cols) > 0:
    print(f"   Found {len(categorical_cols)} categorical features")
    
    # Dictionary to store label encoders for future use
    label_encoders = {}
    
    for col in categorical_cols:
        print(f"\n   {col}:")
        print(f"      Unique values: {df[col].unique()}")
        
        # Use Label Encoding for binary and ordinal variables
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        # Show encoding mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"      Encoding: {mapping}")
        
        # Drop original column and rename encoded column
        df = df.drop(columns=[col])
        df = df.rename(columns={col + '_encoded': col})
    
    print(f"\n   ✓ All categorical variables encoded")
else:
    print("   No categorical variables to encode")

# =============================================================================
# FEATURE SUMMARY
# =============================================================================

print("\n7. Final Feature Summary...")
print(f"   Total features (excluding target): {df.shape[1] - 1}")
print(f"   Feature list:")

X_columns = [col for col in df.columns if col != TARGET]
for i, col in enumerate(X_columns, 1):
    print(f"      {i}. {col}")

print(f"\n   Target variable: {TARGET}")
print(f"   Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

# =============================================================================
# PREPARE FEATURES AND TARGET
# =============================================================================

print("\n8. Preparing Features and Target...")

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"   Features (X): {X.shape}")
print(f"   Target (y): {y.shape}")
print(f"   Target distribution:")
print(f"      Mean: {y.mean():.4f}")
print(f"      Std: {y.std():.4f}")
print(f"      Min: {y.min():.4f}")
print(f"      Max: {y.max():.4f}")

# =============================================================================
# TRAIN-TEST SPLIT
# =============================================================================

print(f"\n9. Splitting Data (Train: {int((1-TEST_SIZE)*100)}% | Test: {int(TEST_SIZE*100)}%)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=None  # Regression task, no stratification
)

print(f"   ✓ Train set: {X_train.shape[0]} samples")
print(f"   ✓ Test set: {X_test.shape[0]} samples")

# =============================================================================
# FEATURE SCALING (ONLY NUMERICAL FEATURES)
# =============================================================================

print("\n10. Applying Feature Scaling (StandardScaler)...")

# Identify numerical and categorical features
# Categorical/Binary features (already encoded 0/1) - should NOT be scaled
categorical_features = ['Gender', 'Company Type', 'WFH Setup Available']

# Numerical features - should be scaled
numerical_features = [col for col in X.columns if col not in categorical_features]

print(f"\n   Numerical features to scale ({len(numerical_features)}):")
for feat in numerical_features:
    print(f"      - {feat}")

print(f"\n   Categorical features (not scaled) ({len(categorical_features)}):")
for feat in categorical_features:
    print(f"      - {feat}")

# Scale only numerical features
scaler = StandardScaler()
X_train_numerical_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_numerical_scaled = scaler.transform(X_test[numerical_features])

# Convert back to DataFrame
X_train_numerical_scaled = pd.DataFrame(
    X_train_numerical_scaled, 
    columns=numerical_features, 
    index=X_train.index
)
X_test_numerical_scaled = pd.DataFrame(
    X_test_numerical_scaled, 
    columns=numerical_features, 
    index=X_test.index
)

# Combine scaled numerical features with unscaled categorical features
X_train_scaled = pd.concat([X_train_numerical_scaled, X_train[categorical_features]], axis=1)
X_test_scaled = pd.concat([X_test_numerical_scaled, X_test[categorical_features]], axis=1)

# Reorder columns to match original order
X_train_scaled = X_train_scaled[X.columns]
X_test_scaled = X_test_scaled[X.columns]

print(f"\n   ✓ Scaled {len(numerical_features)} numerical features (mean=0, std=1)")
print(f"   ✓ Kept {len(categorical_features)} categorical features unscaled (0/1)")
print(f"   ✓ Scaler fitted on training data only")

# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

print("\n11. Saving Processed Data...")

# Save unscaled data
X_train.to_csv('processed_data/X_train.csv', index=False)
X_test.to_csv('processed_data/X_test.csv', index=False)
y_train.to_csv('processed_data/y_train.csv', index=False)
y_test.to_csv('processed_data/y_test.csv', index=False)
print("   ✓ Saved unscaled data:")
print("      - processed_data/X_train.csv")
print("      - processed_data/X_test.csv")
print("      - processed_data/y_train.csv")
print("      - processed_data/y_test.csv")

# Save scaled data
X_train_scaled.to_csv('processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('processed_data/X_test_scaled.csv', index=False)
print("\n   ✓ Saved scaled data:")
print("      - processed_data/X_train_scaled.csv")
print("      - processed_data/X_test_scaled.csv")

# Save complete processed dataset (for reference)
df_processed = pd.concat([X, y], axis=1)
df_processed.to_csv('processed_data/processed_dataset.csv', index=False)
print("\n   ✓ Saved complete processed dataset:")
print("      - processed_data/processed_dataset.csv")

# Save scaler for future use
import joblib
joblib.dump(scaler, 'processed_data/scaler.pkl')
print("\n   ✓ Saved scaler object:")
print("      - processed_data/scaler.pkl")

# =============================================================================
# FEATURE STATISTICS
# =============================================================================

print("\n12. Feature Statistics (Training Set)...")
print("\n" + "="*80)
print("FEATURE STATISTICS")
print("="*80)
print(X_train.describe())

# =============================================================================
# CORRELATION ANALYSIS (FINAL FEATURES)
# =============================================================================

print("\n" + "="*80)
print("FINAL FEATURE CORRELATIONS WITH TARGET")
print("="*80)

# Combine train features with target for correlation analysis
train_with_target = pd.concat([X_train, y_train], axis=1)
final_correlations = train_with_target.corr()[TARGET].sort_values(ascending=False)

print("\nFeature Correlations (sorted by absolute value):")
final_correlations_sorted = final_correlations.drop(TARGET).reindex(
    final_correlations.drop(TARGET).abs().sort_values(ascending=False).index
)

for feature, corr in final_correlations_sorted.items():
    abs_corr = abs(corr)
    bar_length = int(abs_corr * 40)
    bar = '█' * bar_length
    print(f"{feature:.<40} {corr:>7.4f} {bar}")

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n" + "="*80)
print("PREPROCESSING SUMMARY")
print("="*80)

print(f"\n✓ Original Dataset:")
print(f"   - Rows: {original_shape[0]}")
print(f"   - Columns: {original_shape[1]}")

print(f"\n✓ Processed Dataset:")
print(f"   - Rows: {len(df)} (retained {len(df)/original_shape[0]*100:.1f}%)")
print(f"   - Features: {df.shape[1] - 1}")
print(f"   - Target: 1")

print(f"\n✓ Features Removed: {len(features_removed)}")
for feature in features_removed:
    print(f"   - {feature}")

print(f"\n✓ Features Retained: {len(X.columns)}")
for feature in X.columns:
    print(f"   - {feature}")

print(f"\n✓ Data Splits:")
print(f"   - Training: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   - Testing: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n✓ All processed files saved to 'processed_data/' directory")

print("\n" + "="*80)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*80)

print("\nNext Steps:")
print("1. Use X_train_scaled.csv and y_train.csv for model training")
print("2. Use X_test_scaled.csv and y_test.csv for model evaluation")
print("3. Load scaler.pkl when preprocessing new data")
print("4. The dataset is now ready for ML algorithms!")

print("\n" + "="*80)
