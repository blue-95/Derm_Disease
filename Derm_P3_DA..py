import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

#1... DATA PREPROCESSING AND CLEANING

df = pd.read_csv('dermatology_database_1.csv')
df.head()

#Handling inconsistency in 'age' column
df['age'] = df['age'].replace('?', np.nan)
df['age'].sample(20)

df['age'] = pd.to_numeric(df['age'], errors = 'coerce') #Converting age to numeric
df['age'] = df['age'].fillna(df['age'].median())
df['age'].sample(20)

#Checking for duplicates
print('Duplicate Rows : ', df.duplicated().sum())


#Chekcing age outliers(Analysing Descriptive statistics of age):
print('Age stats:\n ', df['age'].describe())


#USing quantile functions to find the IQR & Identifying outliers(if any) using 1.5 IQR rule
q1  = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = (q3-q1)


#Identifying Outliers using IQR and 1.5 IQR rules
lower_threshold , upper_threshold = q1-(1.5*iqr) , q3 + (1.5*iqr)
print(f"Lower Outlier Threshold : {lower_threshold} \nUpper Outlier Threshold : {upper_threshold}")

outliers = df[(df['age'] < lower_threshold) | (df['age'] > upper_threshold)]
if len(outliers) == 0:
    print("No Outliers detected")
else:
    print(f"Found {len(outliers)} outliers")

#Save the Cleaned Dataset
df.to_csv('dermatology_cleaned.csv', index = True)

#2... STATISTICAL ANALYSIS
df = pd.read_csv('dermatology_cleaned.csv')
#Descriptive Statistics
pd.set_option('display.max_columns', None)
df.describe()


#Dividing the Clinical and Histopathlogical Features
clinical_features = [
    'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon',
    'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement',
    'knee_and_elbow_involvement', 'scalp_involvement'
]

histo_epidermal_features = [
    'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis',
    'clubbing_rete_ridges', 'elongation_rete_ridges',
    'thinning_suprapapillary_epidermis', 'spongiform_pustule',
    'munro_microabcess', 'focal_hypergranulosis',
    'disappearance_granular_layer', 'vacuolisation_damage_basal_layer',
    'spongiosis', 'saw_tooth_appearance_retes', 'follicular_horn_plug',
    'perifollicular_parakeratosis', 'melanin_incontinence'
]

histo_dermal_features = [
    'eosinophils_infiltrate', 'PNL_infiltrate',
    'fibrosis_papillary_dermis', 'inflammatory_mononuclear_infiltrate',
    'band_like_infiltrate'
]

additional_features = ['family_history', 'age']
all_features = clinical_features + histo_epidermal_features + histo_dermal_features + additional_features


#Box Plot for Clinical Features
plt.figure(figsize = (15,8))
sns.boxplot(data = df[clinical_features], palette = 'viridis', medianprops = dict(color = 'black',linewidth = 5))
plt.title("Box Plot of Clinical Features(0-3)",fontsize = 20)
plt.xticks(rotation=0)
plt.ylabel('Score (0-3)')
plt.xlabel('Clinical Feature')
plt.tight_layout()
plt.show()


#Box Plot for Epidermal Histopathological Features
plt.figure(figsize=(45, 10))
sns.boxplot(data=df[histo_epidermal_features], palette='plasma',medianprops = dict(color = 'black',linewidth = 5))
plt.title('Box Plots of Epidermal Histopathological Features (Scores 0-3)', fontsize = 20)
plt.ylabel('Score (0-3)')
plt.xlabel('Histopathological Feature',fontsize = 15)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#Box Plot for dermal Histopathological Features
plt.figure(figsize=(45, 10))
sns.boxplot(data=df[histo_dermal_features], palette='plasma',medianprops = dict(color = 'black',linewidth = 5))
plt.title('Box Plots of Dermal Histopathological Features (Scores 0-3)', fontsize = 20)
plt.ylabel('Score (0-3)')
plt.xlabel('Histopathological Feature',fontsize = 15)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


#%%
class_means = df.groupby('class').mean()
print(class_means)

print("Class distribution:")
class_counts = df['class'].value_counts().sort_index()

class_names = {1: 'Psoriasis', 2: 'Seborrheic Dermatitis', 3: 'Lichen Planus',
               4: 'Pityriasis Rosea', 5: 'Chronic Dermatitis', 6: 'Pityriasis Rubra Pilaris'}

for class_id, count in class_counts.items():
    print(f"Class {class_id} ({class_names[class_id]}): {count} samples ({count/len(df)*100:.1f}%)")

all_features = clinical_features + histo_epidermal_features + histo_dermal_features + additional_features

#Create composite scores,Feature Engineering
df['clinical_score'] = df[clinical_features].sum(axis=1)
df['epidermal_score'] = df[histo_epidermal_features].sum(axis=1)
df['dermal_score'] = df[histo_dermal_features].sum(axis=1)
df['total_symptom_score'] = df['clinical_score'] + df['epidermal_score'] + df['dermal_score']

# Severity ratios
df['clinical_to_histo_ratio'] = df['clinical_score'] / (df['epidermal_score'] + df['dermal_score'] + 0.1)  # +0.1 to avoid division by zero

# Feature importance analysis using individual statistical tests

from scipy.stats import f_oneway

# Test each feature individually
feature_discrimination = []
for feature in all_features:
    groups = []
    for class_id in range(1, 7):
        class_data = df[df['class'] == class_id][feature].dropna()
        if len(class_data) > 0:
            groups.append(class_data.values)

    if len(groups) > 1:
        try:
            stat, pval = f_oneway(*groups)
            feature_discrimination.append((feature, stat, pval))
        except:
            continue

# Sort by F-statistic (higher = more discriminative)
feature_discrimination.sort(key=lambda x: x[1], reverse=True)

print("Top 10 most discriminative features:")
for i, (feature, f_stat, p_val) in enumerate(feature_discrimination[:10]):
    print(f"{i+1}. {feature}: F={f_stat:.2f}, p={p_val:.2e}")

p={p_val:.2e}")
#%%
#Correlation matrix:
correlation_matrix = df[all_features].corr(method='spearman')

# Visualize correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Spearman Correlation Matrix of Features')
plt.savefig('correlation_matrix_features_only.png')
plt.show()

print("Top 10 feature correlations (absolute values):\n", correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()[:20])


#Preparing data for PCA(Standardizing features)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X = df[all_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)


explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
print(cumulative_variance)

# Scree plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
plt.axhline(y=0.92, color='k', linestyle='--', label='92% threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.legend()
plt.tight_layout()
plt.show()


n_components_92 = np.argmax(cumulative_variance >= 0.92) + 1
print(f"Number of components for 92% variance: {n_components_92}")

pca_92 = PCA(n_components=n_components_92)
X_pca_92 = pca_92.fit_transform(X_scaled)
X_pca_92

# Feature importance analysis using individual statistical tests
from scipy.stats import f_oneway

feature_pvalues = {}
for feature in all_features:
    groups = [df[df['class'] == i][feature].values for i in range(1, 7)]
    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]
    if len(groups) > 1:
        stat, pval = f_oneway(*groups)
        feature_pvalues[feature] = pval

# Sort by p-value (most discriminative first)
sorted_features = sorted(feature_pvalues.items(), key=lambda x: x[1])
print("Top 10 most discriminative features:")
for feat, pval in sorted_features[:10]:
    print(f"{feat}: p-value = {pval:.2e}")


plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca_92[:, 0], X_pca_92[:, 1], c=df['class'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Components Colored by Class(1-6)')
plt.colorbar(scatter, label='Class (1-6)')
plt.savefig('pca_first_two_components.png')
plt.show()

#Employing Kruskal-wallis test,
from scipy.stats import kruskal

for i in range(n_components_92):
    groups = [X_pca_92[df['class'] == j, i] for j in range(1, 7)]
    if all(len(g) > 0 for g in groups):
        kw_result = kruskal(*groups)
        print(f"Kruskal-Wallis for PC{i+1}: H-statistic = {kw_result.statistic:.2f}, p-value = {kw_result.pvalue:.4f}")


contingency_table = pd.crosstab(df['family_history'], df['class'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square for family_history vs class: Chi2 = {chi2:.2f}, p-value = {p:.4f}, dof = {dof}")
contingency_table.index = ['No Family History', 'Family History']
class_map = {1: 'Psoriasis', 2: 'Seborrheic Dermatitis', 3: 'Lichen Planus',
             4: 'Pityriasis Rosea', 5: 'Chronic Dermatitis', 6: 'Pityriasis Rubra Pilaris'}
contingency_table.columns = [class_map[i] for i in contingency_table.columns]
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Contingency Table: Family History vs Disease Class')
plt.xlabel('Disease')
plt.ylabel('Family History')
plt.savefig('family_history_contingency.png')
plt.show()


