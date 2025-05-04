# First, install required packages if not already installed
# Uncomment and run these lines if needed
# !pip install sdv
# !pip install matplotlib seaborn pandas numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check SDV version and import correctly
try:
    # For newer SDV versions (0.17.0+)
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    print("Using newer SDV version with CTGANSynthesizer")
    use_new_sdv = True
except ImportError:
    try:
        # For older SDV versions
        from sdv.tabular.ctgan import CTGAN
        print("Using older SDV version with CTGAN")
        use_new_sdv = False
    except ImportError:
        raise ImportError("SDV library not found. Please install it using: pip install sdv")

# Load the preprocessed dataset
print("Loading original dataset...")
real_df = pd.read_csv('preprocessed_dataset.csv')

# Display basic info about the dataset
print(f"Original dataset shape: {real_df.shape}")
print("\nColumn types:")
print(real_df.dtypes)
print("\nSample data:")
print(real_df.head())

# Separate categorical and numerical columns
categorical_columns = ['Domain', 'Level', 'Skill', 'Recommended Resource 1', 
                      'Recommended Resource 2', 'Project']
numerical_columns = ['Month', 'domain_encoded', 'level_encoded', 'skill_encoded', 
                    'resource1_encoded', 'resource2_encoded', 'project_encoded']

# Initialize model based on SDV version
print("\nInitializing model...")
if use_new_sdv:
    # Create metadata for the newer SDV version
    print("Creating metadata for the dataset...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_df)
    
    # Update column types if needed
    for col in categorical_columns:
        metadata.update_column(column_name=col, sdtype='categorical')
    
    # For numerical columns that are actually categorical IDs
    for col in ['domain_encoded', 'level_encoded', 'skill_encoded', 
                'resource1_encoded', 'resource2_encoded', 'project_encoded']:
        metadata.update_column(column_name=col, sdtype='categorical')
    
    # Month is truly numerical
    metadata.update_column(column_name='Month', sdtype='numerical')
    
    # For newer SDV versions - pass the metadata
    model = CTGANSynthesizer(
        metadata=metadata,
        epochs=500,
        batch_size=500,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        embedding_dim=128,
        verbose=True
    )
else:
    # For older SDV versions
    model = CTGAN(
        epochs=500,
        batch_size=500,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        embedding_dim=128,
        verbose=True
    )

# Train the model
print("\nTraining model on the dataset...")
model.fit(real_df)

# Generate synthetic data equal to number of original rows
print("\nGenerating synthetic data...")
synthetic_df = model.sample(len(real_df))

# Save to CSV
synthetic_df.to_csv('synthetic_dataset.csv', index=False)
print("Synthetic data saved to 'synthetic_dataset.csv'")

# Validation: Compare distributions of key columns
def plot_distribution_comparison(real_df, synthetic_df, column, categorical=False):
    plt.figure(figsize=(12, 6))
    
    if categorical:
        # For categorical columns, use countplots
        plt.subplot(1, 2, 1)
        sns.countplot(y=real_df[column], order=real_df[column].value_counts().index[:20])
        plt.title(f"Real: {column} (Top 20)")
        plt.tight_layout()
        
        plt.subplot(1, 2, 2)
        sns.countplot(y=synthetic_df[column], order=synthetic_df[column].value_counts().index[:20])
        plt.title(f"Synthetic: {column} (Top 20)")
    else:
        # For numerical columns, use KDE plots
        plt.subplot(1, 2, 1)
        sns.histplot(real_df[column], kde=True)
        plt.title(f"Real: {column}")
        
        plt.subplot(1, 2, 2)
        sns.histplot(synthetic_df[column], kde=True)
        plt.title(f"Synthetic: {column}")
    
    plt.tight_layout()
    plt.savefig(f"comparison_{column}.png")
    print(f"Saved distribution comparison for {column}")

print("\nGenerating comparison visualizations...")
# Plot some categorical distributions
for column in ['Domain', 'Level'][:2]:  # Limit to prevent too many plots
    plot_distribution_comparison(real_df, synthetic_df, column, categorical=True)

# Plot some numerical distributions
for column in ['Month', 'domain_encoded'][:2]:
    plot_distribution_comparison(real_df, synthetic_df, column, categorical=False)

# Additional validation: Check if the synthetic data maintains relationships between columns
print("\nValidating relationships between columns...")

# Example: Check Domain-Level relationship
def compare_relationship(real_df, synthetic_df, col1, col2):
    # Create cross-tabulations
    real_crosstab = pd.crosstab(real_df[col1], real_df[col2], normalize='index')
    synth_crosstab = pd.crosstab(synthetic_df[col1], synthetic_df[col2], normalize='index')
    
    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    sns.heatmap(real_crosstab, annot=False, cmap='viridis', ax=axes[0])
    axes[0].set_title(f'Real Data: {col1} vs {col2}')
    
    sns.heatmap(synth_crosstab, annot=False, cmap='viridis', ax=axes[1])
    axes[1].set_title(f'Synthetic Data: {col1} vs {col2}')
    
    plt.tight_layout()
    plt.savefig(f"relationship_{col1}_{col2}.png")
    print(f"Saved relationship comparison for {col1} vs {col2}")

# Compare a few key relationships
compare_relationship(real_df, synthetic_df, 'Domain', 'Level')
compare_relationship(real_df, synthetic_df, 'Level', 'Month')

# Validate coherence between text columns and their encoded versions
def validate_encoding_coherence(df, text_col, encoded_col):
    # Check if each unique text value maps to a consistent encoded value
    mapping = df.groupby(text_col)[encoded_col].nunique()
    inconsistent = mapping[mapping > 1]
    
    if len(inconsistent) > 0:
        print(f"\nInconsistent mappings found between {text_col} and {encoded_col}:")
        print(inconsistent)
    else:
        print(f"\nEncoding is consistent between {text_col} and {encoded_col}")

# Check encoding consistency in synthetic data
print("\nValidating encoding coherence in synthetic data...")
validate_encoding_coherence(synthetic_df, 'Domain', 'domain_encoded')
validate_encoding_coherence(synthetic_df, 'Level', 'level_encoded')
validate_encoding_coherence(synthetic_df, 'Skill', 'skill_encoded')

# Advanced analysis: Learning path trajectory comparison
print("\nComparing learning path trajectories...")

def analyze_learning_path(df, domain):
    domain_data = df[df['Domain'] == domain]
    path = domain_data.sort_values('Month')[['Month', 'Level', 'Skill']]
    return path

# Compare paths for a few domains
domains_to_check = real_df['Domain'].value_counts().index[:3]  # Top 3 domains

for domain in domains_to_check:
    real_path = analyze_learning_path(real_df, domain)
    synthetic_path = analyze_learning_path(synthetic_df, domain)
    
    print(f"\nLearning path for {domain}:")
    print(f"Real data has {len(real_path)} entries")
    print(f"Synthetic data has {len(synthetic_path)} entries")
    
    # Optional: Save path comparison to files
    real_path.to_csv(f"real_path_{domain.replace(' ', '_')}.csv", index=False)
    synthetic_path.to_csv(f"synthetic_path_{domain.replace(' ', '_')}.csv", index=False)

print("\nAnalysis complete! The synthetic data has been generated and validated.")