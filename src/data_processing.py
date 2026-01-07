"""
Data Processing Module for Drug Interaction Checker
Handles data loading, cleaning, and preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class DrugDataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_sample_data(self, n_samples=1000):
        """
        Create sample drug interaction data for educational purposes
        """
        np.random.seed(42)
        
        # Common drug names for simulation
        drugs = ['Aspirin', 'Warfarin', 'Metformin', 'Lisinopril', 'Atorvastatin',
                'Omeprazole', 'Amlodipine', 'Simvastatin', 'Levothyroxine', 'Azithromycin',
                'Amoxicillin', 'Hydrochlorothiazide', 'Gabapentin', 'Sertraline', 'Ibuprofen']
        
        data = []
        for _ in range(n_samples):
            drug1 = np.random.choice(drugs)
            drug2 = np.random.choice([d for d in drugs if d != drug1])
            
            # Simulate interaction logic
            high_risk_pairs = [('Warfarin', 'Aspirin'), ('Warfarin', 'Ibuprofen')]
            moderate_risk_pairs = [('Metformin', 'Lisinopril'), ('Atorvastatin', 'Simvastatin')]
            
            pair = tuple(sorted([drug1, drug2]))
            
            if pair in high_risk_pairs or tuple(reversed(pair)) in high_risk_pairs:
                interaction = 1
                severity = 'High'
            elif pair in moderate_risk_pairs or tuple(reversed(pair)) in moderate_risk_pairs:
                interaction = 1
                severity = 'Moderate'
            else:
                interaction = np.random.choice([0, 1], p=[0.7, 0.3])
                severity = np.random.choice(['Low', 'Moderate'], p=[0.6, 0.4]) if interaction else 'None'
            
            data.append({
                'drug1': drug1,
                'drug2': drug2,
                'interaction': interaction,
                'severity': severity,
                'drug1_class': np.random.choice(['Analgesic', 'Anticoagulant', 'Antidiabetic', 'Antihypertensive', 'Statin']),
                'drug2_class': np.random.choice(['Analgesic', 'Anticoagulant', 'Antidiabetic', 'Antihypertensive', 'Statin'])
            })
        
        return pd.DataFrame(data)
    
    def load_data(self, file_path=None):
        """
        Load drug interaction data
        """
        if file_path is None:
            # Create sample data if no file provided
            self.df = self.create_sample_data()
            print("Created sample dataset with {} records".format(len(self.df)))
        else:
            self.df = pd.read_csv(file_path)
            print("Loaded dataset with {} records".format(len(self.df)))
        
        return self.df
    
    def clean_data(self):
        """
        Clean and preprocess the data
        """
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {initial_count - len(self.df)} duplicate records")
        
        # Handle missing values
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            self.df = self.df.dropna()
            print(f"Removed {missing_count} records with missing values")
        
        return self.df
    
    def encode_features(self):
        """
        Encode categorical features for ML models
        """
        categorical_cols = ['drug1', 'drug2', 'drug1_class', 'drug2_class']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        # Encode severity
        if 'severity' in self.df.columns:
            severity_map = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
            self.df['severity_encoded'] = self.df['severity'].map(severity_map)
        
        return self.df
    
    def prepare_features(self):
        """
        Prepare feature matrix for ML models
        """
        # Only use drug and drug class features for interaction prediction
        # Exclude severity_encoded as it's the target for severity classification
        feature_cols = ['drug1_encoded', 'drug2_encoded', 'drug1_class_encoded', 'drug2_class_encoded']
        X = self.df[feature_cols]
        y_interaction = self.df['interaction']
        y_severity = self.df['severity_encoded']
        
        return X, y_interaction, y_severity
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        X, y_interaction, y_severity = self.prepare_features()
        
        X_train, X_test, y_int_train, y_int_test, y_sev_train, y_sev_test = train_test_split(
            X, y_interaction, y_severity, test_size=test_size, random_state=random_state, stratify=y_interaction
        )
        
        return X_train, X_test, y_int_train, y_int_test, y_sev_train, y_sev_test
    
    def get_drug_alternatives(self, drug_name, severity='High'):
        """
        Get alternative drugs for high-severity interactions
        """
        alternatives = {
            'Warfarin': ['Rivaroxaban', 'Apixaban', 'Dabigatran'],
            'Aspirin': ['Acetaminophen', 'Celecoxib'],
            'Ibuprofen': ['Acetaminophen', 'Naproxen'],
            'Metformin': ['Glipizide', 'Glyburide'],
            'Atorvastatin': ['Rosuvastatin', 'Pravastatin']
        }
        
        return alternatives.get(drug_name, ['Consult healthcare provider for alternatives'])
    
    def encode_drug_for_prediction(self, drug_name, drug_class=None):
        """
        Encode a drug name for prediction using fitted encoders
        """
        try:
            # Try to encode using fitted encoder
            if 'drug1' in self.label_encoders:
                # Check if drug is in the encoder's classes
                if drug_name in self.label_encoders['drug1'].classes_:
                    drug_encoded = self.label_encoders['drug1'].transform([drug_name])[0]
                else:
                    # Use hash for unknown drugs (fallback)
                    drug_encoded = hash(drug_name) % len(self.label_encoders['drug1'].classes_)
            else:
                drug_encoded = hash(drug_name) % 10
            
            # Encode drug class
            if drug_class and 'drug1_class' in self.label_encoders:
                if drug_class in self.label_encoders['drug1_class'].classes_:
                    class_encoded = self.label_encoders['drug1_class'].transform([drug_class])[0]
                else:
                    class_encoded = hash(drug_class) % len(self.label_encoders['drug1_class'].classes_)
            else:
                class_encoded = hash(drug_name + "_class") % 5
            
            return drug_encoded, class_encoded
            
        except Exception as e:
            # Fallback to hash-based encoding
            return hash(drug_name) % 10, hash(drug_name + "_class") % 5