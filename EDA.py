"""
Exploratory Data Analysis (EDA) for Employee Burnout Dataset
Dataset: enriched_employee_dataset(1).csv
Focus: Understanding factors contributing to employee burnout in corporate environments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the dataset
print("="*80)
print("EMPLOYEE BURNOUT ANALYSIS - EXPLORATORY DATA ANALYSIS")
print("="*80)
print("\n1. Loading Dataset...")

df = pd.read_csv('Dataset/enriched_employee_dataset(1).csv')
print(f"✓ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")

# =============================================================================
# SECTION 1: BASIC DATA OVERVIEW
# =============================================================================
print("\n" + "="*80)
print("2. BASIC DATA OVERVIEW")
print("="*80)

print("\n2.1 First Few Rows:")
print(df.head())

print("\n2.2 Dataset Info:")
print(df.info())

print("\n2.3 Dataset Shape:")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n2.4 Column Names:")
print(df.columns.tolist())

# =============================================================================
# SECTION 2: MISSING VALUES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("3. MISSING VALUES ANALYSIS")
print("="*80)

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Percentage': missing_percentage.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

print("\nColumns with Missing Values:")
print(missing_df.to_string(index=False))

# Visualize missing values
if len(missing_df) > 0:
    plt.figure(figsize=(12, 6))
    plt.bar(missing_df['Column'], missing_df['Percentage'])
    plt.xlabel('Column Name')
    plt.ylabel('Missing Percentage (%)')
    plt.title('Missing Values by Column')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('plots/missing_values.png', dpi=300, bbox_inches='tight')
    print("\n✓ Missing values visualization saved as 'plots/missing_values.png'")
    plt.close()

# =============================================================================
# SECTION 3: STATISTICAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("4. STATISTICAL SUMMARY")
print("="*80)

print("\n4.1 Numerical Features:")
print(df.describe())

print("\n4.2 Categorical Features:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# =============================================================================
# SECTION 4: TARGET VARIABLE ANALYSIS (BURN RATE)
# =============================================================================
print("\n" + "="*80)
print("5. BURN RATE (TARGET VARIABLE) ANALYSIS")
print("="*80)

if 'Burn Rate' in df.columns:
    print("\n5.1 Burn Rate Statistics:")
    print(df['Burn Rate'].describe())
    
    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(df['Burn Rate'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Burn Rate')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Burn Rate')
    axes[0].axvline(df['Burn Rate'].mean(), color='red', linestyle='--', label=f'Mean: {df["Burn Rate"].mean():.2f}')
    axes[0].legend()
    
    # Box plot
    axes[1].boxplot(df['Burn Rate'].dropna(), vert=True)
    axes[1].set_ylabel('Burn Rate')
    axes[1].set_title('Burn Rate Box Plot')
    
    plt.tight_layout()
    plt.savefig('plots/burn_rate_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Burn rate distribution saved as 'plots/burn_rate_distribution.png'")
    plt.close()

# =============================================================================
# SECTION 5: WORK-LIFE BALANCE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("6. WORK-LIFE BALANCE FACTORS")
print("="*80)

# Key metrics
work_life_cols = ['Work Hours per Week', 'Sleep Hours', 'Work-Life Balance Score']
print("\n6.1 Key Metrics:")
for col in work_life_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")

# Visualize work-life balance factors
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

if 'Work Hours per Week' in df.columns:
    axes[0, 0].hist(df['Work Hours per Week'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 0].set_xlabel('Work Hours per Week')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Work Hours per Week')
    axes[0, 0].axvline(40, color='green', linestyle='--', label='Standard 40h')
    axes[0, 0].legend()

if 'Sleep Hours' in df.columns:
    axes[0, 1].hist(df['Sleep Hours'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Sleep Hours')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Sleep Hours')
    axes[0, 1].axvline(7, color='green', linestyle='--', label='Recommended 7h')
    axes[0, 1].legend()

if 'Work-Life Balance Score' in df.columns:
    axes[1, 0].hist(df['Work-Life Balance Score'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 0].set_xlabel('Work-Life Balance Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Work-Life Balance Score')

if 'Mental Fatigue Score' in df.columns:
    axes[1, 1].hist(df['Mental Fatigue Score'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='salmon')
    axes[1, 1].set_xlabel('Mental Fatigue Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Mental Fatigue Score')

plt.tight_layout()
plt.savefig('plots/work_life_balance_factors.png', dpi=300, bbox_inches='tight')
print("\n✓ Work-life balance factors visualization saved as 'plots/work_life_balance_factors.png'")
plt.close()

# =============================================================================
# SECTION 6: WORKPLACE SUPPORT FACTORS
# =============================================================================
print("\n" + "="*80)
print("7. WORKPLACE SUPPORT FACTORS")
print("="*80)

support_cols = ['Manager Support Score', 'Recognition Frequency', 'Team Size']
print("\n7.1 Support Metrics:")
for col in support_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")

# Visualize support factors
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

if 'Manager Support Score' in df.columns:
    axes[0].hist(df['Manager Support Score'].dropna(), bins=10, edgecolor='black', alpha=0.7, color='purple')
    axes[0].set_xlabel('Manager Support Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Manager Support Score Distribution')

if 'Recognition Frequency' in df.columns:
    axes[1].hist(df['Recognition Frequency'].dropna(), bins=10, edgecolor='black', alpha=0.7, color='gold')
    axes[1].set_xlabel('Recognition Frequency')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Recognition Frequency Distribution')

if 'Team Size' in df.columns:
    axes[2].hist(df['Team Size'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='teal')
    axes[2].set_xlabel('Team Size')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Team Size Distribution')

plt.tight_layout()
plt.savefig('plots/workplace_support_factors.png', dpi=300, bbox_inches='tight')
print("\n✓ Workplace support factors visualization saved as 'plots/workplace_support_factors.png'")
plt.close()

# =============================================================================
# SECTION 7: CORRELATION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("8. CORRELATION ANALYSIS")
print("="*80)

# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nAnalyzing correlations for {len(numerical_cols)} numerical features")

# Compute correlation matrix
correlation_matrix = df[numerical_cols].corr()

# Correlation with Burn Rate
if 'Burn Rate' in correlation_matrix.columns:
    print("\n8.1 Features Most Correlated with Burn Rate:")
    burn_rate_corr = correlation_matrix['Burn Rate'].sort_values(ascending=False)
    print(burn_rate_corr)
    
    # Plot top correlations with Burn Rate
    plt.figure(figsize=(10, 8))
    burn_rate_corr_sorted = burn_rate_corr.drop('Burn Rate').sort_values()
    plt.barh(range(len(burn_rate_corr_sorted)), burn_rate_corr_sorted.values)
    plt.yticks(range(len(burn_rate_corr_sorted)), burn_rate_corr_sorted.index)
    plt.xlabel('Correlation with Burn Rate')
    plt.title('Feature Correlation with Burn Rate')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('plots/burn_rate_correlations.png', dpi=300, bbox_inches='tight')
    print("\n✓ Burn rate correlations saved as 'plots/burn_rate_correlations.png'")
    plt.close()

# Full correlation heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of All Numerical Features')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Full correlation heatmap saved as 'plots/correlation_heatmap.png'")
plt.close()

# =============================================================================
# SECTION 8: CATEGORICAL VARIABLE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("9. CATEGORICAL VARIABLES ANALYSIS")
print("="*80)

# Gender analysis
if 'Gender' in df.columns and 'Burn Rate' in df.columns:
    print("\n9.1 Burn Rate by Gender:")
    gender_burnrate = df.groupby('Gender')['Burn Rate'].agg(['mean', 'median', 'std', 'count'])
    print(gender_burnrate)

# Company Type analysis
if 'Company Type' in df.columns and 'Burn Rate' in df.columns:
    print("\n9.2 Burn Rate by Company Type:")
    company_burnrate = df.groupby('Company Type')['Burn Rate'].agg(['mean', 'median', 'std', 'count'])
    print(company_burnrate)

# WFH Setup analysis
if 'WFH Setup Available' in df.columns and 'Burn Rate' in df.columns:
    print("\n9.3 Burn Rate by WFH Setup:")
    wfh_burnrate = df.groupby('WFH Setup Available')['Burn Rate'].agg(['mean', 'median', 'std', 'count'])
    print(wfh_burnrate)

# Create visualizations for categorical variables
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

if 'Gender' in df.columns and 'Burn Rate' in df.columns:
    df.boxplot(column='Burn Rate', by='Gender', ax=axes[0, 0])
    axes[0, 0].set_title('Burn Rate by Gender')
    axes[0, 0].set_xlabel('Gender')
    axes[0, 0].set_ylabel('Burn Rate')

if 'Company Type' in df.columns and 'Burn Rate' in df.columns:
    df.boxplot(column='Burn Rate', by='Company Type', ax=axes[0, 1])
    axes[0, 1].set_title('Burn Rate by Company Type')
    axes[0, 1].set_xlabel('Company Type')
    axes[0, 1].set_ylabel('Burn Rate')

if 'WFH Setup Available' in df.columns and 'Burn Rate' in df.columns:
    df.boxplot(column='Burn Rate', by='WFH Setup Available', ax=axes[1, 0])
    axes[1, 0].set_title('Burn Rate by WFH Setup')
    axes[1, 0].set_xlabel('WFH Setup Available')
    axes[1, 0].set_ylabel('Burn Rate')

if 'Designation' in df.columns and 'Burn Rate' in df.columns:
    df.boxplot(column='Burn Rate', by='Designation', ax=axes[1, 1])
    axes[1, 1].set_title('Burn Rate by Designation')
    axes[1, 1].set_xlabel('Designation')
    axes[1, 1].set_ylabel('Burn Rate')

plt.suptitle('')  # Remove the automatic title
plt.tight_layout()
plt.savefig('plots/categorical_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Categorical analysis visualization saved as 'plots/categorical_analysis.png'")
plt.close()

# =============================================================================
# SECTION 9: WORKLOAD AND PRESSURE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("10. WORKLOAD AND PRESSURE ANALYSIS")
print("="*80)

pressure_cols = ['Deadline Pressure Score', 'Resource Allocation', 'Work Hours per Week']
print("\n10.1 Pressure Metrics:")
for col in pressure_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")

# Scatter plots for workload factors vs Burn Rate
if 'Burn Rate' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    if 'Work Hours per Week' in df.columns:
        axes[0, 0].scatter(df['Work Hours per Week'], df['Burn Rate'], alpha=0.5, s=10)
        axes[0, 0].set_xlabel('Work Hours per Week')
        axes[0, 0].set_ylabel('Burn Rate')
        axes[0, 0].set_title('Work Hours vs Burn Rate')
        axes[0, 0].grid(True, alpha=0.3)
    
    if 'Sleep Hours' in df.columns:
        axes[0, 1].scatter(df['Sleep Hours'], df['Burn Rate'], alpha=0.5, s=10, color='blue')
        axes[0, 1].set_xlabel('Sleep Hours')
        axes[0, 1].set_ylabel('Burn Rate')
        axes[0, 1].set_title('Sleep Hours vs Burn Rate')
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'Deadline Pressure Score' in df.columns:
        axes[1, 0].scatter(df['Deadline Pressure Score'], df['Burn Rate'], alpha=0.5, s=10, color='red')
        axes[1, 0].set_xlabel('Deadline Pressure Score')
        axes[1, 0].set_ylabel('Burn Rate')
        axes[1, 0].set_title('Deadline Pressure vs Burn Rate')
        axes[1, 0].grid(True, alpha=0.3)
    
    if 'Mental Fatigue Score' in df.columns:
        axes[1, 1].scatter(df['Mental Fatigue Score'], df['Burn Rate'], alpha=0.5, s=10, color='purple')
        axes[1, 1].set_xlabel('Mental Fatigue Score')
        axes[1, 1].set_ylabel('Burn Rate')
        axes[1, 1].set_title('Mental Fatigue vs Burn Rate')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/workload_vs_burnrate.png', dpi=300, bbox_inches='tight')
    print("\n✓ Workload vs burn rate visualization saved as 'plots/workload_vs_burnrate.png'")
    plt.close()

# =============================================================================
# SECTION 10: TIME-BASED ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("11. TIME-BASED ANALYSIS")
print("="*80)

if 'Date of Joining' in df.columns:
    # Convert to datetime
    df['Date of Joining'] = pd.to_datetime(df['Date of Joining'])
    df['Joining Year'] = df['Date of Joining'].dt.year
    df['Joining Month'] = df['Date of Joining'].dt.month
    
    print("\n11.1 Employees by Joining Year:")
    print(df['Joining Year'].value_counts().sort_index())
    
    if 'Burn Rate' in df.columns:
        print("\n11.2 Average Burn Rate by Years in Company:")
        years_burnrate = df.groupby('Years in Company')['Burn Rate'].mean().sort_index()
        print(years_burnrate)
        
        plt.figure(figsize=(12, 6))
        years_burnrate.plot(kind='line', marker='o')
        plt.xlabel('Years in Company')
        plt.ylabel('Average Burn Rate')
        plt.title('Average Burn Rate by Years in Company')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/tenure_vs_burnrate.png', dpi=300, bbox_inches='tight')
        print("\n✓ Tenure vs burn rate visualization saved as 'plots/tenure_vs_burnrate.png'")
        plt.close()

# =============================================================================
# SECTION 11: KEY INSIGHTS AND SUMMARY
# =============================================================================
print("\n" + "="*80)
print("12. KEY INSIGHTS AND SUMMARY")
print("="*80)

print("\n12.1 Dataset Summary:")
print(f"• Total Employees: {len(df):,}")
print(f"• Number of Features: {len(df.columns)}")
print(f"• Time Period: {df['Date of Joining'].min().strftime('%Y-%m-%d')} to {df['Date of Joining'].max().strftime('%Y-%m-%d')}")

if 'Burn Rate' in df.columns:
    print(f"\n12.2 Burn Rate Insights:")
    print(f"• Average Burn Rate: {df['Burn Rate'].mean():.3f}")
    print(f"• Median Burn Rate: {df['Burn Rate'].median():.3f}")
    print(f"• High Burnout (>0.6): {(df['Burn Rate'] > 0.6).sum():,} employees ({(df['Burn Rate'] > 0.6).sum()/len(df)*100:.1f}%)")
    print(f"• Low Burnout (<0.3): {(df['Burn Rate'] < 0.3).sum():,} employees ({(df['Burn Rate'] < 0.3).sum()/len(df)*100:.1f}%)")

print("\n12.3 Work Environment Insights:")
if 'Work Hours per Week' in df.columns:
    print(f"• Average Work Hours: {df['Work Hours per Week'].mean():.1f} hours/week")
    print(f"• Employees Working >50h/week: {(df['Work Hours per Week'] > 50).sum():,} ({(df['Work Hours per Week'] > 50).sum()/len(df)*100:.1f}%)")

if 'Sleep Hours' in df.columns:
    print(f"• Average Sleep Hours: {df['Sleep Hours'].mean():.1f} hours/night")
    print(f"• Employees with <6h Sleep: {(df['Sleep Hours'] < 6).sum():,} ({(df['Sleep Hours'] < 6).sum()/len(df)*100:.1f}%)")

if 'Work-Life Balance Score' in df.columns:
    print(f"• Average Work-Life Balance Score: {df['Work-Life Balance Score'].mean():.2f}")

print("\n12.4 Support System Insights:")
if 'Manager Support Score' in df.columns:
    print(f"• Average Manager Support Score: {df['Manager Support Score'].mean():.2f}")
    print(f"• Low Manager Support (<2): {(df['Manager Support Score'] < 2).sum():,} employees")

if 'Recognition Frequency' in df.columns:
    print(f"• Average Recognition Frequency: {df['Recognition Frequency'].mean():.2f}")
    print(f"• No Recognition (0): {(df['Recognition Frequency'] == 0).sum():,} employees")

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Visualizations:")
print("1. plots/missing_values.png - Missing data analysis")
print("2. plots/burn_rate_distribution.png - Distribution of burnout rates")
print("3. plots/work_life_balance_factors.png - Work-life balance metrics")
print("4. plots/workplace_support_factors.png - Support system analysis")
print("5. plots/burn_rate_correlations.png - Feature correlations with burnout")
print("6. plots/correlation_heatmap.png - Complete correlation matrix")
print("7. plots/categorical_analysis.png - Burnout by categorical variables")
print("8. plots/workload_vs_burnrate.png - Workload factors vs burnout")
print("9. plots/tenure_vs_burnrate.png - Company tenure vs burnout")
print("\nAll visualizations saved in the 'plots/' directory!")
print("="*80)
