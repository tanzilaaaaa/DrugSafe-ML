"""
Advanced Feature Engineering for Drug Interaction Checker
Implements feature creation, selection, and augmentation techniques
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedFeatureEngineer:
    def __init__(self):
        self.feature_selector = None
        self.poly_features = None
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.selected_features = []
        
    def create_interaction_features(self, df):
        """
        Create drug interaction features
        """
        print("Creating interaction features...")
        
        # Drug pair frequency features
        drug_pairs = df.groupby(['drug1', 'drug2']).size().reset_index(name='pair_frequency')
        df = df.merge(drug_pairs, on=['drug1', 'drug2'], how='left')
        
        # Drug frequency features
        drug1_freq = df['drug1'].value_counts().to_dict()
        drug2_freq = df['drug2'].value_counts().to_dict()
        df['drug1_frequency'] = df['drug1'].map(drug1_freq)
        df['drug2_frequency'] = df['drug2'].map(drug2_freq)
        
        # Class combination features
        df['class_combination'] = df['drug1_class'] + '_' + df['drug2_class']
        class_interaction_rate = df.groupby('class_combination')['interaction'].mean().to_dict()
        df['class_interaction_rate'] = df['class_combination'].map(class_interaction_rate)
        
        # Risk score based on historical data
        df['risk_score'] = (df['pair_frequency'] * df['class_interaction_rate']).fillna(0)
        
        # Binary features for high-risk combinations
        high_risk_pairs = [('Warfarin', 'Aspirin'), ('Warfarin', 'Ibuprofen')]
        df['is_high_risk_pair'] = df.apply(
            lambda row: 1 if (row['drug1'], row['drug2']) in high_risk_pairs or 
                            (row['drug2'], row['drug1']) in high_risk_pairs else 0, axis=1
        )
        
        print(f"Created {len(df.columns) - len(self.feature_names)} new features")
        return df
    
    def create_polynomial_features(self, X, degree=2):
        """
        Create polynomial features
        """
        print(f"Creating polynomial features (degree={degree})...")
        
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = self.poly_features.fit_transform(X)
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = self.poly_features.get_feature_names_out(X.columns)
        else:
            feature_names = self.poly_features.get_feature_names_out()
        
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        print(f"Created {X_poly_df.shape[1]} polynomial features")
        return X_poly_df
    
    def select_features(self, X, y, method='selectkbest', k=10):
        """
        Select best features using various methods
        """
        print(f"Selecting top {k} features using {method}...")
        
        if method == 'selectkbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            if hasattr(X, 'columns'):
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features = X.columns[selected_indices].tolist()
            
        elif method == 'rfe':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            self.feature_selector = RFE(estimator, n_features_to_select=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            
            if hasattr(X, 'columns'):
                selected_indices = self.feature_selector.get_support(indices=True)
                self.selected_features = X.columns[selected_indices].tolist()
        
        print(f"Selected features: {self.selected_features}")
        return X_selected
    
    def apply_dimensionality_reduction(self, X, method='pca', n_components=5):
        """
        Apply dimensionality reduction techniques
        """
        print(f"Applying {method} with {n_components} components...")
        
        if method == 'pca':
            self.pca = PCA(n_components=n_components, random_state=42)
            X_reduced = self.pca.fit_transform(X)
            
            # Print explained variance
            explained_var = self.pca.explained_variance_ratio_
            print(f"Explained variance ratio: {explained_var}")
            print(f"Total explained variance: {explained_var.sum():.4f}")
            
            # Create DataFrame with component names
            component_names = [f'PC{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)
            
            return X_reduced_df
            
        elif method == 'tsne':
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
            X_reduced = tsne.fit_transform(X)
            
            component_names = [f'TSNE{i+1}' for i in range(n_components)]
            X_reduced_df = pd.DataFrame(X_reduced, columns=component_names, index=X.index)
            
            return X_reduced_df
    
    def balance_dataset(self, X, y, method='smote'):
        """
        Balance dataset using various techniques
        """
        print(f"Balancing dataset using {method}...")
        print(f"Original class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'undersampling':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        else:
            print(f"Unknown method: {method}")
            return X, y
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        print(f"Balanced class distribution: {pd.Series(y_balanced).value_counts().to_dict()}")
        return X_balanced, y_balanced
    
    def create_drug_embeddings(self, df):
        """
        Create drug embeddings based on interaction patterns
        """
        print("Creating drug embeddings...")
        
        # Create drug-drug interaction matrix
        drugs = list(set(df['drug1'].tolist() + df['drug2'].tolist()))
        interaction_matrix = np.zeros((len(drugs), len(drugs)))
        
        drug_to_idx = {drug: idx for idx, drug in enumerate(drugs)}
        
        for _, row in df.iterrows():
            idx1 = drug_to_idx[row['drug1']]
            idx2 = drug_to_idx[row['drug2']]
            interaction_matrix[idx1][idx2] = row['interaction']
            interaction_matrix[idx2][idx1] = row['interaction']  # Symmetric
        
        # Apply PCA to create embeddings
        pca_embeddings = PCA(n_components=5, random_state=42)
        drug_embeddings = pca_embeddings.fit_transform(interaction_matrix)
        
        # Create embedding dictionary
        embedding_dict = {drug: embedding for drug, embedding in zip(drugs, drug_embeddings)}
        
        # Add embeddings to dataframe
        for i in range(5):
            df[f'drug1_embed_{i}'] = df['drug1'].map(lambda x: embedding_dict[x][i])
            df[f'drug2_embed_{i}'] = df['drug2'].map(lambda x: embedding_dict[x][i])
        
        print("Drug embeddings created successfully")
        return df
    
    def create_temporal_features(self, df):
        """
        Create temporal features (simulated for educational purposes)
        """
        print("Creating temporal features...")
        
        # Simulate time-based features
        np.random.seed(42)
        df['interaction_trend'] = np.random.normal(0, 1, len(df))
        df['seasonal_factor'] = np.sin(np.random.uniform(0, 2*np.pi, len(df)))
        df['time_since_discovery'] = np.random.exponential(5, len(df))
        
        return df
    
    def analyze_feature_correlations(self, df, target_col='interaction', save_path='plots/feature_correlations.png'):
        """
        Analyze and visualize feature correlations
        """
        print("Analyzing feature correlations...")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Find highly correlated features with target
        target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
        print(f"\nTop features correlated with {target_col}:")
        print(target_corr.head(10))
        
        return corr_matrix
    
    def create_statistical_features(self, df):
        """
        Create statistical features from existing data
        """
        print("Creating statistical features...")
        
        # Group statistics
        drug_stats = df.groupby('drug1')['interaction'].agg(['mean', 'std', 'count']).add_prefix('drug1_')
        df = df.merge(drug_stats, left_on='drug1', right_index=True, how='left')
        
        drug_stats = df.groupby('drug2')['interaction'].agg(['mean', 'std', 'count']).add_prefix('drug2_')
        df = df.merge(drug_stats, left_on='drug2', right_index=True, how='left')
        
        # Class statistics
        class_stats = df.groupby('drug1_class')['interaction'].agg(['mean', 'std']).add_prefix('class1_')
        df = df.merge(class_stats, left_on='drug1_class', right_index=True, how='left')
        
        class_stats = df.groupby('drug2_class')['interaction'].agg(['mean', 'std']).add_prefix('class2_')
        df = df.merge(class_stats, left_on='drug2_class', right_index=True, how='left')
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def feature_engineering_pipeline(self, df, target_col='interaction'):
        """
        Complete feature engineering pipeline
        """
        print("\n" + "="*60)
        print("ADVANCED FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        original_features = len(df.columns)
        
        # Step 1: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 2: Create drug embeddings
        df = self.create_drug_embeddings(df)
        
        # Step 3: Create temporal features
        df = self.create_temporal_features(df)
        
        # Step 4: Create statistical features
        df = self.create_statistical_features(df)
        
        # Step 5: Analyze correlations
        self.analyze_feature_correlations(df, target_col)
        
        final_features = len(df.columns)
        print(f"\nFeature engineering complete!")
        print(f"Original features: {original_features}")
        print(f"Final features: {final_features}")
        print(f"New features created: {final_features - original_features}")
        
        return df