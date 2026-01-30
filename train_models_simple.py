"""
Simple Model Training for Employee Burnout Prediction
Classical Machine Learning Models with Fixed Hyperparameters (No Grid Search)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import joblib
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
CV_FOLDS = 5

# =============================================================================
# LOAD DATA
# =============================================================================

print("="*80)
print("EMPLOYEE BURNOUT PREDICTION - SIMPLE MODEL TRAINING")
print("="*80)

print("\n1. Loading Preprocessed Data...")

# Load scaled data for training
X_train = pd.read_csv('processed_data/X_train_scaled.csv')
X_test = pd.read_csv('processed_data/X_test_scaled.csv')
y_train = pd.read_csv('processed_data/y_train.csv').values.ravel()
y_test = pd.read_csv('processed_data/y_test.csv').values.ravel()

print(f"   ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Handle any remaining NaN values
print("\n   Checking for missing values...")
if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
    print(f"   ⚠ Found missing values, imputing with column means...")
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    print(f"   ✓ Missing values imputed")
else:
    print(f"   ✓ No missing values found")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n   Evaluating {model_name}...")
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Train R²': r2_score(y_train, y_train_pred),
        'Test R²': r2_score(y_test, y_test_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred) * 100,
    }
    
    # Cross-validation score
    print(f"      Running {CV_FOLDS}-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring='r2', n_jobs=1)
    metrics['CV R² Mean'] = cv_scores.mean()
    metrics['CV R² Std'] = cv_scores.std()
    
    # Print results
    print(f"      Train R²: {metrics['Train R²']:.4f} | Test R²: {metrics['Test R²']:.4f}")
    print(f"      Train RMSE: {metrics['Train RMSE']:.4f} | Test RMSE: {metrics['Test RMSE']:.4f}")
    print(f"      Train MAE: {metrics['Train MAE']:.4f} | Test MAE: {metrics['Test MAE']:.4f}")
    print(f"      CV R²: {metrics['CV R² Mean']:.4f} ± {metrics['CV R² Std']:.4f}")
    
    return metrics, y_train_pred, y_test_pred

def save_model(model, model_name):
    """Save trained model to disk"""
    filename = f"models/{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, filename)
    print(f"      ✓ Model saved: {filename}")

# =============================================================================
# DEFINE MODELS WITH FIXED HYPERPARAMETERS
# =============================================================================

print("\n2. Defining Models with Fixed Hyperparameters...")

models = {
    'Linear Regression': LinearRegression(),
    
    'Ridge Regression': Ridge(alpha=10, random_state=RANDOM_STATE),
    
    'Lasso Regression': Lasso(alpha=0.01, random_state=RANDOM_STATE, max_iter=10000),
    
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=10000),
    
    'Decision Tree': DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE
    ),
    
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    
    'SVR': SVR(
        C=100,
        gamma='scale',
        epsilon=0.1,
        kernel='rbf'
    ),
    
    'MLP': MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        random_state=RANDOM_STATE,
        verbose=False
    )
}

print(f"   ✓ Configured {len(models)} models with fixed hyperparameters")

# =============================================================================
# TRAIN AND EVALUATE MODELS
# =============================================================================

print("\n" + "="*80)
print("3. TRAINING AND EVALUATING MODELS")
print("="*80)

results = []
predictions = {}
trained_models = {}

for model_name, model in models.items():
    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print('='*80)
    
    start_time = time.time()
    
    # Train model
    print(f"   Training model...")
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"   ✓ Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    metrics, y_train_pred, y_test_pred = evaluate_model(
        model, X_train, X_test, y_train, y_test, model_name
    )
    metrics['Training Time (s)'] = training_time
    
    # Save results
    results.append(metrics)
    predictions[model_name] = {
        'train': y_train_pred,
        'test': y_test_pred
    }
    trained_models[model_name] = model
    
    # Save model
    save_model(model, model_name)

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("\n" + "="*80)
print("4. RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

# Sort by Test R² (descending)
results_df = results_df.sort_values('Test R²', ascending=False)

# Display results table
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print("\n" + results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('results/model_comparison.csv', index=False)
print("\n✓ Results saved to 'results/model_comparison.csv'")

# Identify best model
best_model_name = results_df.iloc[0]['Model']
best_r2 = results_df.iloc[0]['Test R²']
print(f"\n🏆 Best Model: {best_model_name} (Test R² = {best_r2:.4f})")

# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("5. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

feature_names = X_train.columns

for model_name in ['Decision Tree', 'Random Forest']:
    if model_name in trained_models:
        model = trained_models[model_name]
        importances = model.feature_importances_
        
        # Create DataFrame
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"\n{model_name} - Feature Importance:")
        for idx, row in fi_df.iterrows():
            bar = '█' * int(row['Importance'] * 50)
            print(f"   {row['Feature']:.<40} {row['Importance']:.4f} {bar}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

# 1. Model Comparison - R² Scores
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(results_df))
width = 0.35

train_r2 = results_df['Train R²'].values
test_r2 = results_df['Test R²'].values
model_names = results_df['Model'].values

ax.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='steelblue')
ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Model Comparison - R² Scores', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison_r2.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/model_comparison_r2.png")
plt.close()

# 2. Model Comparison - RMSE
fig, ax = plt.subplots(figsize=(14, 6))
train_rmse = results_df['Train RMSE'].values
test_rmse = results_df['Test RMSE'].values

ax.bar(x - width/2, train_rmse, width, label='Train RMSE', alpha=0.8, color='coral')
ax.bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8, color='lightcoral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('Model Comparison - RMSE (Lower is Better)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison_rmse.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/model_comparison_rmse.png")
plt.close()

# 3. Model Comparison - MAE
fig, ax = plt.subplots(figsize=(14, 6))
train_mae = results_df['Train MAE'].values
test_mae = results_df['Test MAE'].values

ax.bar(x - width/2, train_mae, width, label='Train MAE', alpha=0.8, color='skyblue')
ax.bar(x + width/2, test_mae, width, label='Test MAE', alpha=0.8, color='steelblue')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('Model Comparison - Mean Absolute Error', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_comparison_mae.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/model_comparison_mae.png")
plt.close()

# 4. Prediction vs Actual (Best Model)
best_model_preds = predictions[best_model_name]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training set
axes[0].scatter(y_train, best_model_preds['train'], alpha=0.3, s=10)
axes[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Burn Rate', fontsize=12)
axes[0].set_ylabel('Predicted Burn Rate', fontsize=12)
axes[0].set_title(f'{best_model_name} - Training Set', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, best_model_preds['test'], alpha=0.3, s=10, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Burn Rate', fontsize=12)
axes[1].set_ylabel('Predicted Burn Rate', fontsize=12)
axes[1].set_title(f'{best_model_name} - Test Set (R² = {best_r2:.4f})', 
                  fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/best_model_predictions.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/best_model_predictions.png")
plt.close()

# 5. Residual Analysis (Best Model)
residuals_train = y_train - best_model_preds['train']
residuals_test = y_test - best_model_preds['test']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Residual plot - Train
axes[0, 0].scatter(best_model_preds['train'], residuals_train, alpha=0.3, s=10)
axes[0, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
axes[0, 0].set_ylabel('Residuals', fontsize=12)
axes[0, 0].set_title(f'{best_model_name} - Training Residuals', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residual plot - Test
axes[0, 1].scatter(best_model_preds['test'], residuals_test, alpha=0.3, s=10, color='orange')
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Values', fontsize=12)
axes[0, 1].set_ylabel('Residuals', fontsize=12)
axes[0, 1].set_title(f'{best_model_name} - Test Residuals', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Residual distribution - Train
axes[1, 0].hist(residuals_train, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Training Residuals Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Residual distribution - Test
axes[1, 1].hist(residuals_test, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Test Residuals Distribution', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/residual_analysis.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/residual_analysis.png")
plt.close()

# 6. Feature Importance (Random Forest)
if 'Random Forest' in trained_models:
    rf_model = trained_models['Random Forest']
    importances = rf_model.feature_importances_
    
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df['Feature'], fi_df['Importance'], color='teal', alpha=0.8)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('results/feature_importance_rf.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: results/feature_importance_rf.png")
    plt.close()

# 7. Training Time Comparison
plt.figure(figsize=(12, 6))
training_times = results_df['Training Time (s)'].values
plt.bar(range(len(model_names)), training_times, color='steelblue', alpha=0.8)
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
plt.xlabel('Model', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.title('Model Training Time Comparison', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/training_time_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: results/training_time_comparison.png")
plt.close()

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"\n📊 Summary:")
print(f"   • Models Trained: {len(trained_models)}")
print(f"   • Best Model: {best_model_name}")
print(f"   • Best Test R²: {best_r2:.4f}")
print(f"   • Best Test RMSE: {results_df.iloc[0]['Test RMSE']:.4f}")
print(f"   • Best Test MAE: {results_df.iloc[0]['Test MAE']:.4f}")
print(f"   • Total Training Time: {results_df['Training Time (s)'].sum():.2f} seconds")

print(f"\n📁 Output Files:")
print(f"   Models saved in: models/")
for model_name in trained_models.keys():
    filename = f"{model_name.replace(' ', '_').lower()}.pkl"
    print(f"      • {filename}")

print(f"\n   Results saved in: results/")
print(f"      • model_comparison.csv")
print(f"      • model_comparison_r2.png")
print(f"      • model_comparison_rmse.png")
print(f"      • model_comparison_mae.png")
print(f"      • best_model_predictions.png")
print(f"      • residual_analysis.png")
print(f"      • feature_importance_rf.png")
print(f"      • training_time_comparison.png")

print("\n" + "="*80)
print("🎉 All models trained and evaluated successfully!")
print("="*80)
