"""
World Development Clustering Model - Complete Deployment Package
=================================================================

Based on your Jupyter notebook analysis, this provides:
1. Model performance comparison
2. Recommendation for deployment
3. Complete preprocessing and deployment code
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL PERFORMANCE COMPARISON
# ============================================================================

print("=" * 80)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 80)
print()

comparison_df = pd.DataFrame({
    'Model': ['KMeans', 'Hierarchical', 'DBSCAN', 'Gaussian Mixture'],
    'Silhouette Score': [0.2248, 0.1948, 0.3208, 0.1212],
    'Full Coverage': ['✓', '✓', '✗', '✓'],
    'Interpretability': ['High', 'Medium', 'Low', 'Medium'],
    'Deployment Ready': ['✓', '✓', '✗', '○']
})

print(comparison_df.to_string(index=False))
print()
print("Legend: ✓ = Yes/Good, ✗ = No/Poor, ○ = Acceptable")
print()

print("=" * 80)
print("RECOMMENDATION: KMeans (k=3) ⭐")
print("=" * 80)
print("""
WHY KMeans is the BEST CHOICE for Deployment:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. FULL DATA COVERAGE (Critical for production)
   - Assigns ALL 2704 countries to clusters
   - DBSCAN marked many countries as noise (-1), making it unusable
   
2. BUSINESS INTERPRETABILITY
   - 3 clusters can represent: Developed, Developing, Underdeveloped
   - Easy to explain to stakeholders and end-users
   
3. STABILITY & REPRODUCIBILITY
   - Deterministic results with random_state=42
   - Consistent predictions for production deployment
   
4. REASONABLE PERFORMANCE
   - Silhouette score: 0.2248 (acceptable for real-world clustering)
   - While DBSCAN scored 0.3208, it excluded many data points
   
5. PRODUCTION READINESS
   - Fast inference time
   - Low computational requirements
   - Easy to maintain and update

Note: The grid search found k=2 optimal (score=0.2983), but k=3 provides
      better semantic meaning for development levels.
""")
print()


# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def clean_currency_column(series):
    """Convert currency strings like '$102,000,000' to float."""
    return series.str.replace('$', '', regex=False)\
                 .str.replace(',', '', regex=False)\
                 .astype(float)


def prepare_data(df):
    """
    Complete data preprocessing pipeline matching the notebook.
    
    Steps:
    1. Clean currency columns
    2. Handle missing values
    3. Drop Country column
    4. Apply log transformation to skewed features
    5. Scale features
    
    Returns:
    --------
    X_scaled : np.ndarray
        Preprocessed and scaled features
    df_clean : pd.DataFrame
        Cleaned dataframe with original values
    scaler : StandardScaler
        Fitted scaler for deployment
    feature_names : list
        List of feature names
    """
    df_clean = df.copy()
    
    # Step 1: Clean currency columns
    currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
    for col in currency_cols:
        if df_clean[col].dtype == 'object':
            df_clean[col] = clean_currency_column(df_clean[col])
    
    # Convert Business Tax Rate if it's object
    if df_clean['Business Tax Rate'].dtype == 'object':
        df_clean['Business Tax Rate'] = pd.to_numeric(
            df_clean['Business Tax Rate'].str.replace('%', ''), 
            errors='coerce'
        )
    
    # Step 2: Handle missing values with median imputation
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Number of Records']
    
    imputer = SimpleImputer(strategy='median')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # Step 3: Prepare model dataframe (drop Country)
    df_model = df_clean.drop(columns=['Country'], errors='ignore')
    
    # Drop Number of Records if present
    if 'Number of Records' in df_model.columns:
        df_model = df_model.drop(columns=['Number of Records'])
    
    feature_names = df_model.columns.tolist()
    
    # Step 4: Apply log transformation to skewed features
    log_cols = [
        'GDP',
        'CO2 Emissions',
        'Energy Usage',
        'Health Exp/Capita',
        'Tourism Inbound',
        'Tourism Outbound'
    ]
    
    for col in log_cols:
        if col in df_model.columns:
            df_model[col] = np.log1p(df_model[col])
    
    # Step 5: Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)
    
    return X_scaled, df_clean, scaler, feature_names


# ============================================================================
# DEPLOYMENT-READY MODEL CLASS
# ============================================================================

class WorldDevelopmentClusteringModel:
    """
    Production-ready KMeans clustering model for World Development data.
    
    This model:
    - Handles all preprocessing automatically
    - Provides cluster predictions for countries
    - Can be easily deployed via API or batch processing
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.log_transform_cols = [
            'GDP', 'CO2 Emissions', 'Energy Usage',
            'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound'
        ]
        self.currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
        self.metadata = {
            'model_type': 'KMeans',
            'n_clusters': n_clusters,
            'trained_date': None,
            'version': '1.0',
            'silhouette_score': None
        }
    
    def _clean_currency(self, series):
        """Clean currency format."""
        if series.dtype == 'object':
            return series.str.replace('$', '', regex=False)\
                        .str.replace(',', '', regex=False)\
                        .astype(float)
        return series
    
    def _preprocess(self, df, fit=False):
        """Preprocess data with cleaning, imputation, and transformation."""
        df_clean = df.copy()
        
        # Clean currency columns
        for col in self.currency_cols:
            if col in df_clean.columns:
                df_clean[col] = self._clean_currency(df_clean[col])
        
        # Clean Business Tax Rate
        if 'Business Tax Rate' in df_clean.columns and df_clean['Business Tax Rate'].dtype == 'object':
            df_clean['Business Tax Rate'] = pd.to_numeric(
                df_clean['Business Tax Rate'].str.replace('%', ''), 
                errors='coerce'
            )
        
        # Drop Country and Number of Records
        df_model = df_clean.drop(columns=['Country', 'Number of Records'], errors='ignore')
        
        # Store feature names on first fit
        if fit:
            self.feature_names = df_model.columns.tolist()
        else:
            # Ensure features match training data
            df_model = df_model[self.feature_names]
        
        # Impute missing values
        if fit:
            self.imputer = SimpleImputer(strategy='median')
            df_model = pd.DataFrame(
                self.imputer.fit_transform(df_model),
                columns=df_model.columns
            )
        else:
            df_model = pd.DataFrame(
                self.imputer.transform(df_model),
                columns=df_model.columns
            )
        
        # Log transformation
        for col in self.log_transform_cols:
            if col in df_model.columns:
                df_model[col] = np.log1p(df_model[col])
        
        # Scaling
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(df_model)
        else:
            X_scaled = self.scaler.transform(df_model)
        
        return X_scaled
    
    def fit(self, df):
        """Train the model."""
        X_scaled = self._preprocess(df, fit=True)
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self.model.fit(X_scaled)
        
        self.metadata['trained_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        self.metadata['silhouette_score'] = silhouette_score(X_scaled, self.model.labels_)
        
        return self
    
    def predict(self, df):
        """Predict cluster assignments."""
        if self.model is None:
            raise ValueError("Model not trained! Call fit() first.")
        
        X_scaled = self._preprocess(df, fit=False)
        return self.model.predict(X_scaled)
    
    def fit_predict(self, df):
        """Fit and predict in one step."""
        self.fit(df)
        X_scaled = self._preprocess(df, fit=False)
        return self.model.predict(X_scaled)
    
    def get_cluster_profile(self, df, predictions):
        """Get statistical profile of each cluster."""
        df_result = df.copy()
        df_result['Cluster'] = predictions
        
        profiles = []
        for cluster in sorted(df_result['Cluster'].unique()):
            cluster_data = df_result[df_result['Cluster'] == cluster]
            profile = {
                'Cluster': cluster,
                'Count': len(cluster_data),
                'Percentage': f"{len(cluster_data)/len(df)*100:.1f}%"
            }
            
            # Add key metrics
            key_metrics = ['GDP', 'Life Expectancy Female', 'Internet Usage', 'CO2 Emissions']
            for metric in key_metrics:
                if metric in df.columns:
                    values = df[df_result['Cluster'] == cluster][metric]
                    if values.dtype == 'object':
                        values = clean_currency_column(values)
                    profile[f'{metric}_mean'] = values.mean()
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def save(self, filepath='world_development_model.pkl'):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'feature_names': self.feature_names,
                'log_transform_cols': self.log_transform_cols,
                'currency_cols': self.currency_cols,
                'metadata': self.metadata
            }, f)
        
        print(f"✓ Model saved: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath='world_development_model.pkl'):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            n_clusters=data['metadata']['n_clusters'],
            random_state=42
        )
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.imputer = data['imputer']
        instance.feature_names = data['feature_names']
        instance.log_transform_cols = data['log_transform_cols']
        instance.currency_cols = data['currency_cols']
        instance.metadata = data['metadata']
        
        print(f" Model loaded: {filepath}")
        print(f"  Trained: {instance.metadata['trained_date']}")
        print(f"  Silhouette Score: {instance.metadata['silhouette_score']:.4f}")
        
        return instance


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DEPLOYMENT DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Load data
    print("Step 1: Loading data...")
    df = pd.read_excel('/mnt/user-data/uploads/World_development_mesurement.xlsx')
    print(f"Loaded {len(df)} records")
    print(f"Features: {df.shape[1]-1} (excluding Country)")
    print()
    
    # Train model
    print("Step 2: Training KMeans model...")
    model = WorldDevelopmentClusteringModel(n_clusters=3, random_state=42)
    predictions = model.fit_predict(df)
    print(f"Model trained successfully!")
    print(f"Inertia: {model.model.inertia_:.2f}")
    print(f"Silhouette Score: {model.metadata['silhouette_score']:.4f}")
    print()
    
    # Cluster distribution
    print("Step 3: Cluster distribution...")
    cluster_dist = pd.Series(predictions).value_counts().sort_index()
    print()
    for cluster, count in cluster_dist.items():
        print(f"  Cluster {cluster}: {count:4d} countries ({count/len(df)*100:5.1f}%)")
    print()
    
    # Show sample countries per cluster
    print("Step 4: Sample countries per cluster...")
    df['Cluster'] = predictions
    print()
    for cluster in sorted(df['Cluster'].unique()):
        countries = df[df['Cluster'] == cluster]['Country'].head(5).tolist()
        print(f"  Cluster {cluster}: {', '.join(countries)}, ...")
    print()
    
    # Cluster profiles
    print("Step 5: Cluster profiles...")
    profile = model.get_cluster_profile(df, predictions)
    print()
    print(profile.to_string(index=False))
    print()
    
    # Save model
    print("Step 6: Saving model for deployment...")
    model_path = model.save('/home/claude/world_development_kmeans.pkl')
    print()
    
    # Test loading
    print("Step 7: Testing model loading...")
    loaded_model = WorldDevelopmentClusteringModel.load('/home/claude/world_development_kmeans.pkl')
    print()
    
    # Test prediction
    print("Step 8: Testing prediction on new data...")
    test_predictions = loaded_model.predict(df.head(10))
    print(f"✓ Successfully predicted {len(test_predictions)} samples")
    print(f"  Predictions: {test_predictions}")
    print()
    
    print("=" * 80)
    print(" DEPLOYMENT PACKAGE READY!")
    print("=" * 80)
    print("""
Model File: world_development_kmeans.pkl

Usage in Production:
────────────────────
    # Load the model
    model = WorldDevelopmentClusteringModel.load('world_development_kmeans.pkl')
    
    # Make predictions
    new_data = pd.read_excel('new_countries.xlsx')
    clusters = model.predict(new_data)
    
    # Get cluster profiles
    profile = model.get_cluster_profile(new_data, clusters)

API Deployment Example (FastAPI):
─────────────────────────────────
    from fastapi import FastAPI
    import pandas as pd
    
    app = FastAPI()
    model = WorldDevelopmentClusteringModel.load('world_development_kmeans.pkl')
    
    @app.post("/predict")
    def predict(country_data: dict):
        df = pd.DataFrame([country_data])
        cluster = model.predict(df)[0]
        return {"cluster": int(cluster)}
    """)
