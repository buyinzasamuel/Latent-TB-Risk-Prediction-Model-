# -*- coding: utf-8 -*-
"""Latent TB Risk Prediction Model for Uganda
Using TB Case Detection, Notification, and Symptom Prevalence Data
"""

# Install required packages

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import scipy.cluster.hierarchy as sch

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Create output directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(script_dir, 'tb_plots')):
    os.makedirs(os.path.join(script_dir, 'tb_plots'))

print("="*60)
print("LATENT TB RISK PREDICTION MODEL FOR UGANDA")
print("Using TB Case Detection, Notification & Symptom Data")
print("="*60)

# ============================================
# LOAD THE DATASETS
# ============================================

# Load Case Detection Rate data
cdr_df = pd.read_csv(os.path.join(script_dir, 'TB_Case_Detection_rate_.csv'))
print("\n[DATASET] Case Detection Rate Dataset:")
print(f"Shape: {cdr_df.shape}")
print(f"Columns: {list(cdr_df.columns)}")
print(f"Periods: {cdr_df['Period'].unique()}")
print(f"Locations: {cdr_df['Location'].nunique()} regions")

# Load Case Notification Rate data
cnr_df = pd.read_csv(os.path.join(script_dir, 'TB_Case_Notification_Rate_(Per_100,000_Population).csv'))
print("\n[DATASET] Case Notification Rate Dataset:")
print(f"Shape: {cnr_df.shape}")
print(f"Columns: {list(cnr_df.columns)}")
print(f"Periods: {cnr_df['Period'].unique()}")
print(f"Locations: {cnr_df['Location'].nunique()} regions")

# Load Prevalence of TB Symptoms data
symptoms_df = pd.read_csv(os.path.join(script_dir, 'Prevalence_of_TB_symptoms.csv'))
print("\n[DATASET] TB Symptoms Prevalence Dataset:")
print(f"Shape: {symptoms_df.shape}")
print(f"Columns: {list(symptoms_df.columns)}")
print(f"Period: {symptoms_df['Period'].unique()}")
print(f"Symptoms: {symptoms_df['Category Combo'].nunique()} symptom types")

# Display first few rows
print("\n[INFO] First 5 rows of Case Detection Data:")
print(cdr_df.head())

print("\n[INFO] First 5 rows of Case Notification Data:")
print(cnr_df.head())

print("\n[INFO] First 5 rows of Symptoms Data:")
print(symptoms_df.head())

# ============================================
# DATA CLEANING AND PREPROCESSING
# ============================================

print("\n" + "="*60)
print("DATA CLEANING AND PREPROCESSING")
print("="*60)

# Clean Case Detection Rate data
cdr_clean = cdr_df.copy()
cdr_clean.columns = cdr_clean.columns.str.strip().str.lower().str.replace(' ', '_')
cdr_clean['value'] = pd.to_numeric(cdr_clean['value'], errors='coerce')
cdr_clean = cdr_clean.dropna(subset=['value'])
cdr_clean['period'] = cdr_clean['period'].astype(str)

# Extract quarter and year
cdr_clean['year'] = cdr_clean['period'].str[:4].astype(int)
cdr_clean['quarter'] = cdr_clean['period'].str[5:]

print(f"\n[DONE] Cleaned Case Detection Data: {cdr_clean.shape}")

# Clean Case Notification Rate data
cnr_clean = cnr_df.copy()
cnr_clean.columns = cnr_clean.columns.str.strip().str.lower().str.replace(' ', '_')
cnr_clean['value'] = pd.to_numeric(cnr_clean['value'], errors='coerce')
cnr_clean = cnr_clean.dropna(subset=['value'])
cnr_clean['period'] = cnr_clean['period'].astype(str)
cnr_clean['year'] = cnr_clean['period'].astype(int)

print(f"[DONE] Cleaned Case Notification Data: {cnr_clean.shape}")

# Clean Symptoms Data
symptoms_clean = symptoms_df.copy()
symptoms_clean.columns = symptoms_clean.columns.str.strip().str.lower().str.replace(' ', '_')
symptoms_clean['value'] = pd.to_numeric(symptoms_clean['value'], errors='coerce')
symptoms_clean = symptoms_clean.dropna(subset=['value'])
symptoms_clean['age_group'] = symptoms_clean['age_group'].fillna('All Ages')
symptoms_clean['gender'] = symptoms_clean['gender'].fillna('Both')
symptoms_clean['category_combo'] = symptoms_clean['category_combo'].fillna('General')

print(f"[DONE] Cleaned Symptoms Data: {symptoms_clean.shape}")

# ============================================
# FEATURE ENGINEERING
# ============================================

print("\n[TOOLS] Feature Engineering...")

# Create regional summary from CDR
regional_cdr = cdr_clean.groupby('location').agg({
    'value': ['mean', 'std', 'min', 'max']
}).round(2)
regional_cdr.columns = ['cdr_mean', 'cdr_std', 'cdr_min', 'cdr_max']
regional_cdr = regional_cdr.reset_index()

# Create regional summary from CNR
regional_cnr = cnr_clean.groupby('location').agg({
    'value': ['mean', 'std', 'min', 'max']
}).round(2)
regional_cnr.columns = ['cnr_mean', 'cnr_std', 'cnr_min', 'cnr_max']
regional_cnr = regional_cnr.reset_index()

# Merge regional datasets
regional_data = pd.merge(regional_cdr, regional_cnr, on='location', how='outer')
regional_data = regional_data.fillna(0)

# Calculate detection gap (CDR - CNR) - proxy for under-detection
regional_data['detection_gap'] = regional_data['cdr_mean'] - regional_data['cnr_mean']
regional_data['detection_ratio'] = (regional_data['cdr_mean'] / (regional_data['cnr_mean'] + 0.1)).round(2)

print(f"[DONE] Created regional summary with {len(regional_data)} regions")
print(regional_data.head())

# Create symptom summary by age group
symptom_by_age = symptoms_clean.groupby(['age_group', 'category_combo'])['value'].mean().reset_index()
symptom_pivot = symptom_by_age.pivot(index='age_group', columns='category_combo', values='value').fillna(0)

print("\n[DONE] Created symptom pivot table by age group")
print(symptom_pivot.head())

# Create risk score based on symptoms
symptom_weights = {
    'Cough for 2+ weeks': 3,
    'Cough with Sputum': 2,
    'Cough (any duration)': 1,
    'Fever': 2,
    'Night sweats': 3,
    'Weight loss': 3,
    'Chest pain': 2,
    'Bloodstained sputum': 4
}

def calculate_risk_score(row):
    score = 0
    for symptom, weight in symptom_weights.items():
        if symptom in row.index:
            score += row[symptom] * weight / 100  # Normalize by percentage
    return score

symptom_pivot['risk_score'] = symptom_pivot.apply(calculate_risk_score, axis=1)
print("\n[DONE] Calculated symptom-based risk scores by age group")

# Save processed data
regional_data.to_csv('tb_regional_risk_data.csv', index=False)
symptom_pivot.to_csv('tb_symptom_risk_data.csv')
print("\n[SAVED] Processed data saved")

# ============================================
# PART 1: EXPLORATORY DATA ANALYSIS - 25+ VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("PART 1: EXPLORATORY DATA ANALYSIS - 25+ VISUALIZATIONS")
print("="*60)

# Set color palette
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#28A745', '#6C757D', '#17A2B8', '#FFC107']
sns.set_palette(colors)

# ============================================
# SECTION A: CASE DETECTION RATE VISUALIZATIONS (8 plots)
# ============================================

print("\n[PLOTS] Generating Case Detection Rate Visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
axes = axes.flatten()
plot_idx = 0

# 1. Distribution of Case Detection Rates
axes[plot_idx].hist(cdr_clean['value'], bins=20, color=colors[0], edgecolor='black', alpha=0.7)
axes[plot_idx].axvline(cdr_clean['value'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {cdr_clean['value'].mean():.1f}")
axes[plot_idx].axvline(cdr_clean['value'].median(), color='green', linestyle='--',
                       label=f"Median: {cdr_clean['value'].median():.1f}")
axes[plot_idx].set_xlabel('Case Detection Rate (%)')
axes[plot_idx].set_ylabel('Frequency')
axes[plot_idx].set_title('Distribution of TB Case Detection Rates')
axes[plot_idx].legend()
plot_idx += 1

# 2. Case Detection Rate by Region (Box Plot)
region_order = cdr_clean.groupby('location')['value'].median().sort_values().index
sns.boxplot(x='location', y='value', data=cdr_clean, order=region_order, ax=axes[plot_idx])
axes[plot_idx].set_xlabel('Region')
axes[plot_idx].set_ylabel('Case Detection Rate (%)')
axes[plot_idx].set_title('Case Detection Rate by Region')
axes[plot_idx].tick_params(axis='x', rotation=90)
plot_idx += 1

# 3. Time Series of Case Detection Rates by Region
pivot_cdr = cdr_clean.pivot_table(index='period', columns='location', values='value')
for region in pivot_cdr.columns[:8]:  # Plot first 8 regions to avoid overcrowding
    axes[plot_idx].plot(pivot_cdr.index, pivot_cdr[region], marker='o', label=region, linewidth=2)
axes[plot_idx].set_xlabel('Quarter')
axes[plot_idx].set_ylabel('Case Detection Rate (%)')
axes[plot_idx].set_title('Case Detection Rate Trends (Selected Regions)')
axes[plot_idx].legend(loc='upper right', fontsize=8)
axes[plot_idx].tick_params(axis='x', rotation=45)
plot_idx += 1

# 4. Quarterly Trends (Box Plot)
cdr_clean['quarter_num'] = pd.to_numeric(cdr_clean['quarter'].str[1], errors='coerce')
# Filter data to remove rows with missing quarter info
cdr_box_data = cdr_clean.dropna(subset=['quarter_num'])
sns.boxplot(x='quarter', y='value', data=cdr_box_data, ax=axes[plot_idx])
axes[plot_idx].set_xlabel('Quarter')
axes[plot_idx].set_ylabel('Case Detection Rate (%)')
axes[plot_idx].set_title('Case Detection Rate by Quarter')
plot_idx += 1

# 5. Regional Variability (Bar Chart of Means with Error Bars)
regional_stats = cdr_clean.groupby('location')['value'].agg(['mean', 'std']).sort_values('mean', ascending=False).head(15)
axes[plot_idx].barh(regional_stats.index, regional_stats['mean'], xerr=regional_stats['std'],
                    color=colors[2], alpha=0.7, capsize=3)
axes[plot_idx].set_xlabel('Mean Case Detection Rate (%)')
axes[plot_idx].set_title('Top 15 Regions - Mean Case Detection Rate (with Std Dev)')
plot_idx += 1

# 6. Year-over-Year Change
cdr_clean['year'] = cdr_clean['period'].str[:4].astype(int)
yearly_cdr = cdr_clean.groupby(['location', 'year'])['value'].mean().reset_index()
yearly_pivot = yearly_cdr.pivot(index='location', columns='year', values='value')
yearly_pivot['change_24_25'] = yearly_pivot[2025] - yearly_pivot[2024]
yearly_change = yearly_pivot['change_24_25'].dropna().sort_values()

top_increases = yearly_change.tail(5)
top_decreases = yearly_change.head(5)

axes[plot_idx].barh(range(len(top_increases)), top_increases.values, color='green', alpha=0.7, label='Increases')
axes[plot_idx].barh(range(-len(top_decreases), 0), top_decreases.values, color='red', alpha=0.7, label='Decreases')
axes[plot_idx].set_yticks(range(len(top_increases)))
axes[plot_idx].set_yticklabels(top_increases.index)
axes[plot_idx].set_xlabel('Change in Detection Rate (2024-2025)')
axes[plot_idx].set_title('Regions with Largest Year-over-Year Changes')
axes[plot_idx].legend()
plot_idx += 1

# 7. Heatmap of Case Detection Rates by Region and Quarter
heatmap_data = cdr_clean.pivot_table(index='location', columns='period', values='value')
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[plot_idx], cbar_kws={'label': 'Detection Rate (%)'})
axes[plot_idx].set_title('Case Detection Rate Heatmap: Region × Quarter')
plot_idx += 1

# 8. Correlation between Quarters
quarter_corr = cdr_clean.pivot_table(index='location', columns='quarter', values='value').corr()
sns.heatmap(quarter_corr, annot=True, cmap='RdBu_r', center=0, ax=axes[plot_idx])
axes[plot_idx].set_title('Correlation Between Quarters')
plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/case_detection_visualizations.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SECTION B: CASE NOTIFICATION RATE VISUALIZATIONS (8 plots)
# ============================================

print("\n[PLOTS] Generating Case Notification Rate Visualizations...")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
axes = axes.flatten()
plot_idx = 0

# 9. Distribution of Case Notification Rates
axes[plot_idx].hist(cnr_clean['value'], bins=30, color=colors[1], edgecolor='black', alpha=0.7)
axes[plot_idx].axvline(cnr_clean['value'].mean(), color='red', linestyle='--',
                       label=f"Mean: {cnr_clean['value'].mean():.1f}")
axes[plot_idx].set_xlabel('Case Notification Rate (per 100,000)')
axes[plot_idx].set_ylabel('Frequency')
axes[plot_idx].set_title('Distribution of TB Case Notification Rates')
axes[plot_idx].legend()
plot_idx += 1

# 10. Time Series of National Notification Rate
national_cnr = cnr_clean[cnr_clean['location'] == 'Uganda'].sort_values('year')
axes[plot_idx].plot(national_cnr['year'], national_cnr['value'], marker='o', linewidth=3, color=colors[1])
axes[plot_idx].fill_between(national_cnr['year'], national_cnr['value'], alpha=0.3, color=colors[1])
axes[plot_idx].set_xlabel('Year')
axes[plot_idx].set_ylabel('Notification Rate (per 100,000)')
axes[plot_idx].set_title('National TB Case Notification Rate Trend (2018-2024)')
axes[plot_idx].grid(True, alpha=0.3)
plot_idx += 1

# 11. Notification Rate by Region (Bar Chart)
region_cnr = cnr_clean[cnr_clean['location'] != 'Uganda'].groupby('location')['value'].mean().sort_values(ascending=False)
top_regions = region_cnr.head(10)
bars = axes[plot_idx].bar(range(len(top_regions)), top_regions.values, color=colors[2])
axes[plot_idx].set_xticks(range(len(top_regions)))
axes[plot_idx].set_xticklabels(top_regions.index, rotation=45, ha='right')
axes[plot_idx].set_xlabel('Region')
axes[plot_idx].set_ylabel('Mean Notification Rate')
axes[plot_idx].set_title('Top 10 Regions by TB Notification Rate')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_regions.values)):
    axes[plot_idx].text(i, val + 1, f'{val:.1f}', ha='center', fontsize=9)
plot_idx += 1

# 12. Box Plot of Notification Rates by Year
sns.boxplot(x='year', y='value', data=cnr_clean[cnr_clean['location'] != 'Uganda'], ax=axes[plot_idx])
axes[plot_idx].set_xlabel('Year')
axes[plot_idx].set_ylabel('Notification Rate')
axes[plot_idx].set_title('Distribution of Regional Notification Rates by Year')
plot_idx += 1

# 13. Violin Plot of Notification Rates
sns.violinplot(x='year', y='value', data=cnr_clean[cnr_clean['location'] != 'Uganda'], ax=axes[plot_idx])
axes[plot_idx].set_xlabel('Year')
axes[plot_idx].set_ylabel('Notification Rate')
axes[plot_idx].set_title('Violin Plot: Regional Notification Rate Distribution')
plot_idx += 1

# 14. Regional Variability (Box Plot)
region_order_cnr = cnr_clean[cnr_clean['location'] != 'Uganda'].groupby('location')['value'].median().sort_values().tail(10).index
sns.boxplot(x='value', y='location', data=cnr_clean[cnr_clean['location'].isin(region_order_cnr)], ax=axes[plot_idx])
axes[plot_idx].set_xlabel('Notification Rate')
axes[plot_idx].set_ylabel('Region')
axes[plot_idx].set_title('Top 10 Regions - Notification Rate Distribution')
plot_idx += 1

# 15. Heatmap of Notification Rates by Region and Year
heatmap_cnr = cnr_clean[cnr_clean['location'] != 'Uganda'].pivot_table(index='location', columns='year', values='value')
sns.heatmap(heatmap_cnr, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[plot_idx])
axes[plot_idx].set_title('Notification Rate Heatmap: Region × Year')
plot_idx += 1

# 16. Correlation Matrix of Regions
region_corr = cnr_clean[cnr_clean['location'] != 'Uganda'].pivot_table(index='year', columns='location', values='value').corr()
# Sample a subset of regions for visibility
sample_regions = region_corr.columns[:10]
sns.heatmap(region_corr.loc[sample_regions, sample_regions], annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[plot_idx])
axes[plot_idx].set_title('Correlation Between Regions (Selected)')
plot_idx += 1

# Hide unused subplots
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/notification_rate_visualizations.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SECTION C: SYMPTOMS PREVALENCE VISUALIZATIONS (6 plots)
# ============================================

print("\n[PLOTS] Generating Symptoms Prevalence Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
plot_idx = 0

# 17. Top Symptoms Overall
symptom_avg = symptoms_clean.groupby('category_combo')['value'].mean().sort_values(ascending=False).head(10)
bars = axes[plot_idx].barh(range(len(symptom_avg)), symptom_avg.values, color=colors)
axes[plot_idx].set_yticks(range(len(symptom_avg)))
axes[plot_idx].set_yticklabels(symptom_avg.index)
axes[plot_idx].set_xlabel('Mean Prevalence (%)')
axes[plot_idx].set_title('Top 10 TB Symptoms by Prevalence')
for i, (bar, val) in enumerate(zip(bars, symptom_avg.values)):
    axes[plot_idx].text(val + 0.5, i, f'{val:.1f}%', va='center')
plot_idx += 1

# 18. Symptoms by Age Group
symptom_age = symptoms_clean.pivot_table(index='age_group', columns='category_combo', values='value')
top_symptoms = symptom_avg.index[:5]  # Top 5 symptoms
symptom_age_top = symptom_age[top_symptoms]

symptom_age_top.plot(kind='bar', ax=axes[plot_idx], colormap='viridis')
axes[plot_idx].set_xlabel('Age Group')
axes[plot_idx].set_ylabel('Prevalence (%)')
axes[plot_idx].set_title('Top Symptoms by Age Group')
axes[plot_idx].legend(loc='upper right', fontsize=8)
axes[plot_idx].tick_params(axis='x', rotation=45)
plot_idx += 1

# 19. Symptoms by Gender
symptom_gender = symptoms_clean[symptoms_clean['gender'].isin(['Male', 'Female'])].pivot_table(
    index='category_combo', columns='gender', values='value').fillna(0)
symptom_gender_top = symptom_gender.loc[top_symptoms]

x = np.arange(len(top_symptoms))
width = 0.35
axes[plot_idx].bar(x - width/2, symptom_gender_top['Male'], width, label='Male', color=colors[0])
axes[plot_idx].bar(x + width/2, symptom_gender_top['Female'], width, label='Female', color=colors[1])
axes[plot_idx].set_xlabel('Symptom')
axes[plot_idx].set_ylabel('Prevalence (%)')
axes[plot_idx].set_title('Symptoms by Gender')
axes[plot_idx].set_xticks(x)
axes[plot_idx].set_xticklabels(top_symptoms, rotation=45, ha='right')
axes[plot_idx].legend()
plot_idx += 1

# 20. Heatmap of Symptoms by Age and Gender
symptom_heat = symptoms_clean.pivot_table(index='age_group', columns='category_combo', values='value')
sns.heatmap(symptom_heat, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[plot_idx], cbar_kws={'label': 'Prevalence (%)'})
axes[plot_idx].set_title('Symptom Prevalence Heatmap: Age × Symptom')
plot_idx += 1

# 21. Age Distribution of Key Symptoms
key_symptoms = ['Cough for 2+ weeks', 'Fever', 'Night sweats', 'Weight loss', 'Chest pain']
age_order = ['15 - 24 Years', '25 - 34 Years', '35 - 44', '45 - 54 Years', '55 - 64 Years', 'Above 65 Years']

for symptom in key_symptoms:
    symptom_data = symptoms_clean[symptoms_clean['category_combo'] == symptom]
    symptom_data = symptom_data[symptom_data['age_group'].isin(age_order)]
    axes[plot_idx].plot(symptom_data['age_group'], symptom_data['value'], marker='o', label=symptom, linewidth=2)

axes[plot_idx].set_xlabel('Age Group')
axes[plot_idx].set_ylabel('Prevalence (%)')
axes[plot_idx].set_title('Age Trends for Key Symptoms')
axes[plot_idx].legend(loc='upper left', fontsize=8)
axes[plot_idx].tick_params(axis='x', rotation=45)
plot_idx += 1

# 22. Symptom Risk Score by Age Group
symptom_pivot_sorted = symptom_pivot.sort_values('risk_score', ascending=False)
bars = axes[plot_idx].barh(range(len(symptom_pivot_sorted)), symptom_pivot_sorted['risk_score'], color=colors[2])
axes[plot_idx].set_yticks(range(len(symptom_pivot_sorted)))
axes[plot_idx].set_yticklabels(symptom_pivot_sorted.index)
axes[plot_idx].set_xlabel('Risk Score')
axes[plot_idx].set_title('Symptom-Based TB Risk Score by Age Group')
for i, (bar, val) in enumerate(zip(bars, symptom_pivot_sorted['risk_score'].values)):
    axes[plot_idx].text(val + 0.05, i, f'{val:.2f}', va='center')
plot_idx += 1

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/symptoms_visualizations.png'), dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# SECTION D: COMBINED/ADVANCED VISUALIZATIONS (6+ plots)
# ============================================

print("\n[PLOTS] Generating Combined and Advanced Visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
plot_idx = 0

# 23. CDR vs CNR Scatter Plot
merged_rates = pd.merge(
    cdr_clean.groupby('location')['value'].mean().reset_index().rename(columns={'value': 'cdr'}),
    cnr_clean[cnr_clean['location'] != 'Uganda'].groupby('location')['value'].mean().reset_index().rename(columns={'value': 'cnr'}),
    on='location', how='inner'
)

scatter = axes[plot_idx].scatter(merged_rates['cnr'], merged_rates['cdr'], 
                                  s=100, c=range(len(merged_rates)), cmap='viridis', alpha=0.7)
axes[plot_idx].set_xlabel('Mean Notification Rate')
axes[plot_idx].set_ylabel('Mean Detection Rate')
axes[plot_idx].set_title('Detection Rate vs Notification Rate by Region')

# Add diagonal line
max_val = max(merged_rates[['cdr', 'cnr']].max().max(), 100)
axes[plot_idx].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
axes[plot_idx].legend()

# Add region labels for key points
for _, row in merged_rates.iterrows():
    if row['cdr'] > 100 or row['cnr'] > 80:
        axes[plot_idx].annotate(row['location'][:4], (row['cnr'], row['cdr']), fontsize=8)

plt.colorbar(scatter, ax=axes[plot_idx], label='Region Index')
plot_idx += 1

# 24. Detection Gap Analysis
merged_rates['detection_gap'] = merged_rates['cdr'] - merged_rates['cnr']
merged_rates_sorted = merged_rates.sort_values('detection_gap', ascending=False)

bars = axes[plot_idx].barh(range(len(merged_rates_sorted.head(10))), 
                           merged_rates_sorted.head(10)['detection_gap'].values, 
                           color=['red' if x < 0 else 'green' for x in merged_rates_sorted.head(10)['detection_gap'].values])
axes[plot_idx].set_yticks(range(len(merged_rates_sorted.head(10))))
axes[plot_idx].set_yticklabels(merged_rates_sorted.head(10)['location'].values)
axes[plot_idx].set_xlabel('Detection Gap (CDR - CNR)')
axes[plot_idx].set_title('Top 10 Regions by Detection Gap (Over-detection vs Under-detection)')
axes[plot_idx].axvline(x=0, color='black', linestyle='-', linewidth=1)
plot_idx += 1

# 25. Risk Score vs Detection Rate
age_risk = symptom_pivot[['risk_score']].reset_index()
age_risk.columns = ['age_group', 'risk_score']

# Create age group mapping for merging
age_mapping = {
    '15 - 24 Years': 'Young Adult',
    '25 - 34 Years': 'Adult',
    '35 - 44': 'Middle Age',
    '45 - 54 Years': 'Older Adult',
    '55 - 64 Years': 'Senior',
    'Above 65 Years': 'Elderly'
}
age_risk['age_category'] = age_risk['age_group'].map(age_mapping)

# Simplified bar chart of risk by age
bars = axes[plot_idx].bar(age_risk['age_group'], age_risk['risk_score'], color=colors)
axes[plot_idx].set_xlabel('Age Group')
axes[plot_idx].set_ylabel('Symptom Risk Score')
axes[plot_idx].set_title('TB Symptom Risk Score by Age Group')
axes[plot_idx].tick_params(axis='x', rotation=45)
for bar, val in zip(bars, age_risk['risk_score']):
    axes[plot_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}', ha='center', fontsize=9)
plot_idx += 1

# 26. Regional Performance Quadrant
merged_rates['cdr_category'] = pd.cut(merged_rates['cdr'], bins=[0, 85, 100, 150], labels=['Low', 'Medium', 'High'])
merged_rates['cnr_category'] = pd.cut(merged_rates['cnr'], bins=[0, 50, 80, 150], labels=['Low', 'Medium', 'High'])

category_counts = pd.crosstab(merged_rates['cdr_category'], merged_rates['cnr_category'])
sns.heatmap(category_counts, annot=True, fmt='d', cmap='YlOrRd', ax=axes[plot_idx])
axes[plot_idx].set_xlabel('Notification Rate Category')
axes[plot_idx].set_ylabel('Detection Rate Category')
axes[plot_idx].set_title('Regional Performance Quadrant')
plot_idx += 1

# 27. Temporal Trend Comparison
years = list(range(2018, 2025))
national_trends = cnr_clean[cnr_clean['location'] == 'Uganda'].set_index('year')['value'].to_dict()

regional_trends = {}
for region in ['Acholi', 'Karamoja', 'Teso', 'West Nile']:
    region_data = cnr_clean[(cnr_clean['location'] == region) & (cnr_clean['year'].isin(years))]
    for _, row in region_data.iterrows():
        if region not in regional_trends:
            regional_trends[region] = {}
        regional_trends[region][row['year']] = row['value']

for region, data in regional_trends.items():
    region_years = sorted(data.keys())
    region_values = [data[y] for y in region_years]
    axes[plot_idx].plot(region_years, region_values, marker='o', label=region, linewidth=2)

axes[plot_idx].plot(years, [national_trends.get(y, 0) for y in years], marker='s', 
                    label='National', linewidth=3, color='black', linestyle='--')
axes[plot_idx].set_xlabel('Year')
axes[plot_idx].set_ylabel('Notification Rate')
axes[plot_idx].set_title('Regional vs National Trends')
axes[plot_idx].legend(loc='upper right', fontsize=8)
axes[plot_idx].grid(True, alpha=0.3)
plot_idx += 1

# Hide unused subplot
for i in range(plot_idx, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/combined_visualizations.png'), dpi=150, bbox_inches='tight')
plt.show()

# 28. Interactive 3D Scatter Plot
fig = px.scatter_3d(merged_rates, x='cnr', y='cdr', z='detection_gap',
                    color='detection_gap', text='location',
                    title='3D Visualization: Detection, Notification, and Gap',
                    labels={'cnr': 'Notification Rate', 'cdr': 'Detection Rate', 'detection_gap': 'Detection Gap'})
fig.write_html(os.path.join(script_dir, 'tb_plots/3d_regional_analysis.html'))
print("[DONE] 3D scatter plot saved as HTML")

# 29. Parallel Coordinates Plot
from pandas.plotting import parallel_coordinates

plot_df = merged_rates.copy()
plot_df['performance'] = pd.cut(plot_df['detection_gap'], 
                                 bins=[-100, -20, 20, 100], 
                                 labels=['Under-detected', 'Balanced', 'Over-detected'])

plt.figure(figsize=(15, 8))
parallel_coordinates(plot_df, 'performance', 
                     cols=['cdr', 'cnr', 'detection_gap'],
                     color=[colors[0], colors[1], colors[2]])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Parallel Coordinates: Regional Performance Profiles')
plt.legend()
plt.savefig(os.path.join(script_dir, 'tb_plots/parallel_coordinates.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n[DONE] TOTAL VISUALIZATIONS: 29 plots generated")
print("[SAVED] All plots saved to 'tb_plots/' directory")

# ============================================
# PART 2: UNSUPERVISED LEARNING
# ============================================

print("\n" + "="*60)
print("PART 2: UNSUPERVISED LEARNING")
print("="*60)

# Prepare features for clustering
cluster_features = ['cdr_mean', 'cnr_mean', 'detection_gap', 'detection_ratio', 
                    'cdr_std', 'cnr_std']

X_cluster = regional_data[cluster_features].copy()

# Handle any missing values
X_cluster = X_cluster.fillna(0)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print(f"\n[ANALYSIS] Using {len(cluster_features)} features for clustering")
print(f"Data shape: {X_scaled.shape}")

# ============================================
# A. K-MEANS CLUSTERING
# ============================================

print("\n" + "-"*40)
print("A. K-MEANS CLUSTERING")
print("-"*40)

# Find optimal k
inertias = []
silhouette_scores = []
K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score by k')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/kmeans_optimization.png'), dpi=150)
plt.show()

# Choose optimal k
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✅ Optimal k: {optimal_k} (silhouette score = {max(silhouette_scores):.3f})")

# Fit final K-Means
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
regional_data['kmeans_cluster'] = final_kmeans.fit_predict(X_scaled)

# Analyze clusters
print("\n📊 Cluster Distribution:")
print(regional_data['kmeans_cluster'].value_counts().sort_index())

# Cluster profiles
cluster_profiles = regional_data.groupby('kmeans_cluster')[cluster_features].mean()
print("\n[ANALYSIS] Cluster Profiles (mean values):")
print(cluster_profiles.round(2))

# Visualize clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=regional_data['kmeans_cluster'], 
                     cmap='viridis', s=100, alpha=0.7, edgecolors='black')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title(f'K-Means Clusters of Ugandan Regions (k={optimal_k})')
plt.colorbar(scatter, label='Cluster')

# Add region labels
for i, txt in enumerate(regional_data['location']):
    plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

plt.savefig(os.path.join(script_dir, 'tb_plots/kmeans_clusters.png'), dpi=150)
plt.show()

# Characterize clusters
print("\n📌 Cluster Characterization:")
for cluster_id in range(optimal_k):
    cluster_data = regional_data[regional_data['kmeans_cluster'] == cluster_id]
    print(f"\nCluster {cluster_id} (n={len(cluster_data)} regions):")
    print(f"  Regions: {', '.join(cluster_data['location'].tolist())}")
    print(f"  Mean Detection Rate: {cluster_data['cdr_mean'].mean():.1f}")
    print(f"  Mean Notification Rate: {cluster_data['cnr_mean'].mean():.1f}")
    print(f"  Mean Detection Gap: {cluster_data['detection_gap'].mean():.1f}")

# ============================================
# B. HIERARCHICAL CLUSTERING
# ============================================

print("\n" + "-"*40)
print("B. HIERARCHICAL CLUSTERING")
print("-"*40)

# Compute linkage matrix
linkage_matrix = sch.linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram = sch.dendrogram(linkage_matrix, labels=regional_data['location'].values,
                           leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram of Ugandan Regions')
plt.xlabel('Region')
plt.ylabel('Distance')
plt.axhline(y=5, color='red', linestyle='--', label='Cut at distance=5')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/dendrogram.png'), dpi=150)
plt.show()

# ============================================
# C. GAUSSIAN MIXTURE MODEL
# ============================================

print("\n" + "-"*40)
print("C. GAUSSIAN MIXTURE MODEL")
print("-"*40)

gmm = GaussianMixture(n_components=optimal_k, random_state=42)
regional_data['gmm_cluster'] = gmm.fit_predict(X_scaled)
cluster_probs = gmm.predict_proba(X_scaled)
regional_data['gmm_confidence'] = cluster_probs.max(axis=1)

print(f"\n📊 GMM Cluster Distribution:")
print(regional_data['gmm_cluster'].value_counts().sort_index())
print(f"\n📈 Average confidence: {regional_data['gmm_confidence'].mean():.3f}")

# ============================================
# D. DBSCAN FOR ANOMALY DETECTION
# ============================================

print("\n" + "-"*40)
print("D. DBSCAN ANOMALY DETECTION")
print("-"*40)

dbscan = DBSCAN(eps=1.5, min_samples=2)
regional_data['dbscan_cluster'] = dbscan.fit_predict(X_scaled)

n_outliers = (regional_data['dbscan_cluster'] == -1).sum()
print(f"\n[ANALYSIS] DBSCAN Results:")
print(f"  Outliers detected: {n_outliers} ({n_outliers/len(regional_data)*100:.1f}%)")

if n_outliers > 0:
    outliers = regional_data[regional_data['dbscan_cluster'] == -1]
    print(f"  Outlier regions: {', '.join(outliers['location'].tolist())}")

# Save clustering results
regional_data.to_csv('tb_regional_clusters.csv', index=False)
print("\n[SAVED] Clustering results saved to 'tb_regional_clusters.csv'")

# ============================================
# PART 3: SEMI-SUPERVISED LEARNING
# ============================================

print("\n" + "="*60)
print("PART 3: SEMI-SUPERVISED LEARNING")
print("="*60)

"""
Scenario: We only have "risk labels" for some regions (e.g., based on expert assessment)
We want to propagate these labels to unlabeled regions
"""

# Create synthetic labels based on detection gap
regional_data['risk_category'] = pd.cut(regional_data['detection_gap'], 
                                         bins=[-100, -20, 20, 100],
                                         labels=['High Risk (Under-detected)', 
                                                 'Moderate Risk', 
                                                 'Low Risk (Over-detected)'])

# Encode as numeric
le_risk = LabelEncoder()
regional_data['risk_encoded'] = le_risk.fit_transform(regional_data['risk_category'])

# Prepare features
feature_cols = ['cdr_mean', 'cnr_mean', 'cdr_std', 'cnr_std', 'detection_ratio']
X_all = regional_data[feature_cols].fillna(0).values
y_all = regional_data['risk_encoded'].values

# Standardize
scaler_ss = StandardScaler()
X_scaled_all = scaler_ss.fit_transform(X_all)

# Create semi-supervised scenario: only 40% have labels
np.random.seed(42)
labeled_idx = np.random.choice(len(X_all), size=int(0.4 * len(X_all)), replace=False)

# Create labels with -1 for unlabeled
y_mixed = np.full(len(X_all), -1)
y_mixed[labeled_idx] = y_all[labeled_idx]

print(f"\n[ANALYSIS] Semi-supervised setup:")
print(f"  Total regions: {len(X_all)}")
print(f"  Labeled regions: {len(labeled_idx)} ({len(labeled_idx)/len(X_all)*100:.1f}%)")
print(f"  Unlabeled regions: {len(X_all) - len(labeled_idx)}")

# ============================================
# A. LABEL PROPAGATION
# ============================================

print("\n" + "-"*40)
print("A. LABEL PROPAGATION")

label_prop = LabelPropagation(kernel='knn', n_neighbors=3, max_iter=1000)
label_prop.fit(X_scaled_all, y_mixed)

y_prop_pred = label_prop.transduction_
prop_accuracy = accuracy_score(y_all[labeled_idx], y_prop_pred[labeled_idx])
print(f"\n[DONE] Label Propagation Accuracy (on labeled data): {prop_accuracy:.3f}")

# ============================================
# B. LABEL SPREADING
# ============================================

print("\n" + "-"*40)
print("B. LABEL SPREADING")

label_spread = LabelSpreading(kernel='knn', n_neighbors=3, alpha=0.8)
label_spread.fit(X_scaled_all, y_mixed)

y_spread_pred = label_spread.transduction_
spread_accuracy = accuracy_score(y_all[labeled_idx], y_spread_pred[labeled_idx])
print(f"\n✅ Label Spreading Accuracy (on labeled data): {spread_accuracy:.3f}")

# ============================================
# C. COMPARISON WITH SUPERVISED BASELINE
# ============================================

print("\n" + "-"*40)
print("C. SUPERVISED BASELINE")

X_labeled = X_scaled_all[labeled_idx]
y_labeled = y_all[labeled_idx]

supervised = RandomForestClassifier(n_estimators=50, random_state=42)
supervised.fit(X_labeled, y_labeled)
supervised_pred = supervised.predict(X_scaled_all[labeled_idx])
supervised_acc = accuracy_score(y_labeled, supervised_pred)

print(f"\n[DONE] Supervised Accuracy (labeled only): {supervised_acc:.3f}")

print(f"\n[ANALYSIS] Performance Comparison:")
print(f"  Supervised (labeled only): {supervised_acc:.3f}")
print(f"  Label Propagation: {prop_accuracy:.3f}")
print(f"  Label Spreading: {spread_accuracy:.3f}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# True labels
true_display = np.full(len(X_all), np.nan)
true_display[labeled_idx] = y_all[labeled_idx]
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=true_display, 
                           cmap='viridis', s=100, alpha=0.7)
axes[0].set_title('True Labels (labeled only)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0])

# Label Propagation
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_prop_pred, 
                           cmap='viridis', s=100, alpha=0.7)
axes[1].set_title('Label Propagation Predictions')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[1])

# Label Spreading
scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_spread_pred, 
                           cmap='viridis', s=100, alpha=0.7)
axes[2].set_title('Label Spreading Predictions')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')
plt.colorbar(scatter3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/semi_supervised_results.png'), dpi=150)
plt.show()

# ============================================
# PART 4: WEAK SUPERVISION
# ============================================

print("\n" + "="*60)
print("PART 4: WEAK SUPERVISION - HEURISTIC LABELING")
print("="*60)

"""
Weak Supervision: Create labels using domain knowledge rules
without requiring manual labeling
"""

def labeling_function_1(row):
    """Rule 1: High detection gap (>20) indicates high risk"""
    if row['detection_gap'] > 20:
        return 2  # High risk
    elif row['detection_gap'] < -20:
        return 0  # Low risk
    return -1

def labeling_function_2(row):
    """Rule 2: Low detection rate (<85) with high notification (>50) indicates high risk"""
    if row['cdr_mean'] < 85 and row['cnr_mean'] > 50:
        return 2
    elif row['cdr_mean'] > 100 and row['cnr_mean'] < 30:
        return 0
    return -1

def labeling_function_3(row):
    """Rule 3: High variability (std > 15) indicates uncertainty/high risk"""
    if row['cdr_std'] > 15 or row['cnr_std'] > 20:
        return 2
    elif row['cdr_std'] < 5 and row['cnr_std'] < 5:
        return 0
    return -1

def labeling_function_4(row):
    """Rule 4: Detection ratio < 1.5 indicates under-detection (high risk)"""
    if row['detection_ratio'] < 1.5:
        return 2
    elif row['detection_ratio'] > 2.5:
        return 0
    return -1

# Apply labeling functions
print("\n[ANALYSIS] Applying weak labeling functions...")

labeling_functions = [labeling_function_1, labeling_function_2, 
                      labeling_function_3, labeling_function_4]

weak_labels = np.zeros((len(regional_data), len(labeling_functions))) - 1

for i, lf in enumerate(labeling_functions):
    for j, (idx, row) in enumerate(regional_data.iterrows()):
        weak_labels[j, i] = lf(row)

# Count votes per region
votes_per_region = (weak_labels != -1).sum(axis=1)
regional_data['weak_votes'] = votes_per_region

print(f"\n[ANALYSIS] Weak labeling statistics:")
print(f"  Regions with any weak label: {(votes_per_region > 0).sum()}/{len(regional_data)} ({(votes_per_region > 0).mean()*100:.1f}%)")
print(f"  Average votes per region: {votes_per_region.mean():.2f}")
print(f"  Max votes: {votes_per_region.max()}")

# Combine weak labels using majority vote
def majority_vote(row_votes):
    valid = row_votes[row_votes != -1]
    if len(valid) == 0:
        return -1
    return np.bincount(valid.astype(int)).argmax()

regional_data['weak_majority'] = np.apply_along_axis(majority_vote, 1, weak_labels)

# Map back to labels
risk_map = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk'}
regional_data['weak_label'] = regional_data['weak_majority'].map(risk_map)

print("\n📊 Weak label distribution:")
print(regional_data['weak_majority'].value_counts().sort_index())
print(regional_data['weak_label'].value_counts())

# Compare with actual risk category (for evaluation)
comparison_mask = regional_data['weak_majority'] != -1
if comparison_mask.sum() > 0:
    weak_acc = accuracy_score(
        regional_data.loc[comparison_mask, 'risk_encoded'],
        regional_data.loc[comparison_mask, 'weak_majority']
    )
    print(f"\n[DONE] Weak supervision accuracy: {weak_acc:.3f}")

# Train a model using weak labels
train_mask = regional_data['weak_majority'] != -1
X_train_weak = X_scaled_all[train_mask]
y_train_weak = regional_data.loc[train_mask, 'weak_majority'].values

weak_model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
weak_model.fit(X_train_weak, y_train_weak)

# Predict on all data
regional_data['weak_model_pred'] = weak_model.predict(X_scaled_all)
weak_model_acc = accuracy_score(regional_data['risk_encoded'], regional_data['weak_model_pred'])
print(f"\n[DONE] Weak supervision model overall accuracy: {weak_model_acc:.3f}")

# Visualize weak label agreement
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Distribution of votes
axes[0].hist(regional_data['weak_votes'], bins=range(1, 6), alpha=0.7, 
             color=colors[0], edgecolor='black', align='left')
axes[0].set_xlabel('Number of Labeling Functions Voting')
axes[0].set_ylabel('Number of Regions')
axes[0].set_title('Distribution of Weak Label Votes')
axes[0].set_xticks(range(1, 5))

# Agreement between labeling functions
agreement_matrix = np.zeros((len(labeling_functions), len(labeling_functions)))
for i in range(len(labeling_functions)):
    for j in range(len(labeling_functions)):
        if i != j:
            mask_i = weak_labels[:, i] != -1
            mask_j = weak_labels[:, j] != -1
            common = mask_i & mask_j
            if common.sum() > 0:
                agreement = (weak_labels[common, i] == weak_labels[common, j]).mean()
                agreement_matrix[i, j] = agreement

sns.heatmap(agreement_matrix, annot=True, cmap='YlGnBu', vmin=0, vmax=1,
            xticklabels=['LF1', 'LF2', 'LF3', 'LF4'],
            yticklabels=['LF1', 'LF2', 'LF3', 'LF4'], ax=axes[1])
axes[1].set_title('Labeling Function Agreement')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/weak_supervision.png'), dpi=150)
plt.show()

# ============================================
# PART 5: POSITIVE-UNLABELED (PU) LEARNING
# ============================================

print("\n" + "="*60)
print("PART 5: POSITIVE-UNLABELED (PU) LEARNING")
print("="*60)

"""
PU Learning: Identify high-risk regions when only some are confirmed positive
"""

# Define positive class: High-risk regions (detection gap > 20 or cdr_mean < 80)
regional_data['is_high_risk'] = ((regional_data['detection_gap'] > 20) | 
                                  (regional_data['cdr_mean'] < 80)).astype(int)

# Only observe 50% of positives
pos_indices = regional_data[regional_data['is_high_risk'] == 1].index
observed_pos = np.random.choice(pos_indices, size=int(0.5 * len(pos_indices)), replace=False)

regional_data['pu_label'] = -1
regional_data.loc[observed_pos, 'pu_label'] = 1

print(f"\n[ANALYSIS] PU Setup:")
print(f"  Total regions: {len(regional_data)}")
print(f"  True high-risk: {regional_data['is_high_risk'].sum()}")
print(f"  Observed positives: {len(observed_pos)}")
print(f"  Hidden positives: {regional_data['is_high_risk'].sum() - len(observed_pos)}")

# Prepare data
X_pu = X_scaled_all
y_pu = regional_data['pu_label'].values

# Two-step PU learning
X_pos = X_pu[y_pu == 1]
X_unlabeled = X_pu[y_pu == -1]

# Find reliable negatives (far from positive centroid)
pos_centroid = X_pos.mean(axis=0)
distances = np.linalg.norm(X_unlabeled - pos_centroid, axis=1)
threshold = np.percentile(distances, 70)  # Top 30% farthest
reliable_neg = X_unlabeled[distances > threshold]

# Train classifier on positives + reliable negatives
X_train_pu = np.vstack([X_pos, reliable_neg])
y_train_pu = np.hstack([np.ones(len(X_pos)), np.zeros(len(reliable_neg))])

pu_model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
pu_model.fit(X_train_pu, y_train_pu)

# Predict on all data
pu_scores = pu_model.predict_proba(X_pu)[:, 1]
regional_data['pu_score'] = pu_scores
regional_data['pu_pred'] = (pu_scores > 0.5).astype(int)

# Evaluate
pu_acc = accuracy_score(regional_data['is_high_risk'], regional_data['pu_pred'])
print(f"\n[DONE] PU Learning Accuracy: {pu_acc:.3f}")

# Precision-Recall
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(regional_data['is_high_risk'], pu_scores)
pr_auc = auc(recall, precision)
print(f"  PR-AUC: {pr_auc:.3f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PU scores by true class
axes[0].hist([regional_data[regional_data['is_high_risk']==0]['pu_score'],
              regional_data[regional_data['is_high_risk']==1]['pu_score']],
             bins=10, label=['Low Risk', 'High Risk'],
             color=[colors[0], colors[1]], alpha=0.7, edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--')
axes[0].set_xlabel('PU Score')
axes[0].set_ylabel('Count')
axes[0].set_title('PU Scores by True Risk Class')
axes[0].legend()

# Precision-Recall Curve
axes[1].plot(recall, precision, color=colors[1], lw=2, label=f'PR-AUC = {pr_auc:.3f}')
axes[1].fill_between(recall, precision, alpha=0.2, color=colors[1])
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Top candidates for screening
top_candidates = regional_data.nlargest(5, 'pu_score')[['location', 'pu_score', 'cdr_mean', 'cnr_mean']]
bars = axes[2].barh(range(len(top_candidates)), top_candidates['pu_score'].values[::-1], color=colors[1])
axes[2].set_yticks(range(len(top_candidates)))
axes[2].set_yticklabels(top_candidates['location'].values[::-1])
axes[2].set_xlabel('PU Score')
axes[2].set_title('Top 5 Regions for Targeted Screening')
axes[2].set_xlim(0, 1)
axes[2].axvline(0.5, color='red', linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_candidates['pu_score'].values[::-1])):
    axes[2].text(val + 0.02, i, f'{val:.2f}', va='center')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'tb_plots/pu_learning.png'), dpi=150)
plt.show()

