import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv('../data/augmented_dataset.csv')  # Update path if necessary

# Encode categorical variables for correlation matrix
df_encoded = df.copy()
for col in ['Domain', 'Level', 'Skill', 'Recommended Resource 1', 'Recommended Resource 2', 'Project']:
    df_encoded[col] = df_encoded[col].astype('category').cat.codes

# Compute correlation matrix
corr_matrix = df_encoded.corr(numeric_only=True)

# Set plot style
sns.set(style="whitegrid")

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# 1. Countplot of Domain
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Domain')
plt.title('Count of Records per Domain')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/count_of_records_per_domain.png')
plt.close()

# 2. Countplot of Level
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Level', order=df['Level'].value_counts().index)
plt.title('Count of Records per Level')
plt.tight_layout()
plt.savefig('plots/count_of_records_per_level.png')
plt.close()

# 3. Countplot of Month
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Month')
plt.title('Distribution over Months')
plt.tight_layout()
plt.savefig('plots/distribution_over_months.png')
plt.close()

# 4. Top 10 most frequent Skills
plt.figure(figsize=(8, 6))
top_skills = df['Skill'].value_counts().nlargest(10)
sns.barplot(x=top_skills.values, y=top_skills.index)
plt.title('Top 10 Skills')
plt.tight_layout()
plt.savefig('plots/top_10_skills.png')
plt.close()

# 5. Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# 6. Countplot of Recommended Resource 1 (Top 10)
plt.figure(figsize=(8, 6))
top_resources = df['Recommended Resource 1'].value_counts().nlargest(10)
sns.barplot(x=top_resources.values, y=top_resources.index)
plt.title("Top 10 Recommended Resources")
plt.tight_layout()
plt.savefig('plots/top_10_recommended_resources.png')
plt.close()
