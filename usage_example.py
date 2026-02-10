import pandas as pd
import sys
sys.path.append('/home/claude')
from deployment_complete import WorldDevelopmentClusteringModel

print("=" * 80)
print("SIMPLE USAGE EXAMPLE")
print("=" * 80)
print()

# ============================================================================
# Example 1: Load and Use Pre-trained Model
# ============================================================================

print("Example 1: Using the pre-trained model")
print("-" * 80)

# Load the model
model = WorldDevelopmentClusteringModel.load('/home/claude/world_development_kmeans.pkl')
print()

# Load original data
df = pd.read_excel('/mnt/user-data/uploads/World_development_mesurement.xlsx')
print(f"✓ Loaded {len(df)} countries from dataset")
print()

# Make predictions
predictions = model.predict(df)
df['Cluster'] = predictions

# Show distribution
print("Cluster Distribution:")
for cluster in sorted(df['Cluster'].unique()):
    count = (df['Cluster'] == cluster).sum()
    print(f"  Cluster {cluster}: {count} countries ({count/len(df)*100:.1f}%)")
print()

# Show sample countries from each cluster
print("Sample Countries per Cluster:")
for cluster in sorted(df['Cluster'].unique()):
    countries = df[df['Cluster'] == cluster]['Country'].head(3).tolist()
    print(f"  Cluster {cluster}: {', '.join(countries)}")
print()

# ============================================================================
# Example 2: Predict for a New Country
# ============================================================================

print("\nExample 2: Predicting cluster for a new country")
print("-" * 80)

# Create sample data for a hypothetical country
new_country = pd.DataFrame({
    'Country': ['Hypothetical Country'],
    'Birth Rate': [0.018],
    'Business Tax Rate': [25.5],
    'CO2 Emissions': [350000],
    'Days to Start Business': [7.0],
    'Ease of Business': [75.0],
    'Energy Usage': [2200],
    'GDP': [450000000000],
    'Health Exp % GDP': [8.5],
    'Health Exp/Capita': [4500],
    'Hours to do Tax': [180],
    'Infant Mortality Rate': [8.5],
    'Internet Usage': [0.68],
    'Lending Interest': [5.2],
    'Life Expectancy Female': [83.5],
    'Life Expectancy Male': [79.2],
    'Mobile Phone Usage': [1.15],
    'Number of Records': [1],
    'Population 0-14': [0.18],
    'Population 15-64': [0.65],
    'Population 65+': [0.17],
    'Population Total': [45000000],
    'Population Urban': [0.82],
    'Tourism Inbound': [12000000],
    'Tourism Outbound': [9000000]
})

print("New Country Data:")
print(f"  Name: {new_country['Country'].values[0]}")
print(f"  GDP: ${new_country['GDP'].values[0]:,.0f}")
print(f"  Population: {new_country['Population Total'].values[0]:,}")
print(f"  Life Expectancy (F): {new_country['Life Expectancy Female'].values[0]} years")
print(f"  Internet Usage: {new_country['Internet Usage'].values[0]*100:.0f}%")
print()

cluster = model.predict(new_country)[0]
cluster_names = {0: "Developed", 1: "Developing", 2: "Emerging"}

print(f"Prediction: Cluster {cluster} ({cluster_names[cluster]})")
print()

# ============================================================================
# Example 3: Compare Multiple Countries
# ============================================================================

print("\nExample 3: Comparing specific countries")
print("-" * 80)

# Select a few countries to compare
countries_to_compare = ['United States', 'China', 'India', 'Brazil', 'Nigeria']
available_countries = [c for c in countries_to_compare if c in df['Country'].values]

if available_countries:
    comparison_df = df[df['Country'].isin(available_countries)][['Country', 'Cluster']].copy()
    
    # Add cluster names
    comparison_df['Cluster_Name'] = comparison_df['Cluster'].map(cluster_names)
    
    print("Country Comparisons:")
    for _, row in comparison_df.iterrows():
        print(f"  {row['Country']:20s} → Cluster {row['Cluster']} ({row['Cluster_Name']})")
    print()

# ============================================================================
# Example 4: Analyze Cluster Characteristics
# ============================================================================

print("\nExample 4: Detailed cluster characteristics")
print("-" * 80)

profile = model.get_cluster_profile(df, predictions)
print()
print(profile.to_string(index=False))
print()

# ============================================================================
# Example 5: Save Results to Excel
# ============================================================================

print("\nExample 5: Saving results")
print("-" * 80)

# Add cluster names to the dataframe
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# Save to Excel
output_file = '/home/claude/clustered_countries.xlsx'
df[['Country', 'Cluster', 'Cluster_Name', 'GDP', 'Population Total']].to_excel(
    output_file, 
    index=False
)
print(f"✓ Results saved to: {output_file}")
print()

print("=" * 80)
print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()

print("""
Next Steps:
-----------
1. Modify the examples above for your specific use case
2. Deploy the model using the API (api_deployment.py)
3. Integrate into your data pipeline
4. Set up monitoring and logging for production use

For more details, see README.md
""")
