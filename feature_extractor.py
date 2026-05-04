import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

class FeatureExtractor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        
        # Complete list of 41 NSL-KDD feature names (as per official documentation)
        self.FEATURE_NAMES = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        
        # Full column list for NSL-KDD (features + label + difficulty)
        # The raw file has 43 columns: 41 features, then label, then difficulty
        self.COLUMN_NAMES = self.FEATURE_NAMES + ['label', 'difficulty']
        
        # Attack mapping: 39 subtypes -> 5 main categories
        self.ATTACK_GROUPS = {
            'normal': 'normal',
            'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
            'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'processtable': 'dos',
            'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
            'mscan': 'probe', 'saint': 'probe',
            'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l', 'multihop': 'r2l',
            'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l', 'warezmaster': 'r2l',
            'sendmail': 'r2l', 'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
            'xlock': 'r2l', 'xsnoop': 'r2l',
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r', 'rootkit': 'u2r',
            'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
        }
        
        self.CLASS_TO_INT = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
    
    def load_dataset(self, filepath):
        """Load dataset with proper column names and drop the 'difficulty' column."""
        df = pd.read_csv(filepath, names=self.COLUMN_NAMES, skipinitialspace=True)
        # Drop the extra 'difficulty' column (last column)
        if 'difficulty' in df.columns:
            df = df.drop('difficulty', axis=1)
        return df
    
    def encode_categorical(self, df, fit=True):
        """Encode categorical features; for test set, map unseen values to 0."""
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            values = df[col].astype(str).str.strip()
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(values)
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                known = set(le.classes_)
                def safe_encode(x):
                    return le.transform([x])[0] if x in known else 0
                df[col] = values.apply(safe_encode)
        return df
    
    def map_labels(self, df):
        """Map attack subtypes to main categories (5 classes)."""
        # Clean label column
        df['label'] = df['label'].astype(str).str.strip()
        df['attack_group'] = df['label'].apply(lambda x: self.ATTACK_GROUPS.get(x, 'normal'))
        df['class_int'] = df['attack_group'].map(self.CLASS_TO_INT)
        return df['class_int'].values
    
    def scale_features(self, X_train, X_test=None):
        """Min-Max normalization."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled
    
    def preprocess(self, train_path, test_path):
        """Complete preprocessing pipeline."""
        # Load
        train_df = self.load_dataset(train_path)
        test_df = self.load_dataset(test_path)
        
        # Encode categorical (fit on training only)
        train_df = self.encode_categorical(train_df, fit=True)
        test_df = self.encode_categorical(test_df, fit=False)
        
        # Features (all feature columns, not including 'label')
        X_train = train_df[self.FEATURE_NAMES].values
        X_test = test_df[self.FEATURE_NAMES].values
        
        # Scale
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Labels
        y_train = self.map_labels(train_df)
        y_test = self.map_labels(test_df)
        
        # Save preprocessors
        self._save_preprocessors()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, self.FEATURE_NAMES
    
    def _save_preprocessors(self):
        os.makedirs('models/preprocessors', exist_ok=True)
        joblib.dump(self.label_encoders, 'models/preprocessors/label_encoders.pkl')
        joblib.dump(self.scaler, 'models/preprocessors/scaler.pkl')
        joblib.dump(self.FEATURE_NAMES, 'models/features.pkl')