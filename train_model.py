# train_model.py
from feature_extractor import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def train_and_evaluate():
    print("=" * 60)
    print("NIDS MODEL TRAINING ON NSL-KDD DATASET")
    print("=" * 60)
    
    # Load and preprocess
    print("\n[1/5] Loading and preprocessing NSL-KDD dataset...")
    extractor = FeatureExtractor()
    X_train, X_test, y_train, y_test, feature_cols = extractor.preprocess(
        train_path='data/KDDTrain+.txt',
        test_path='data/KDDTest+.txt'
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n[2/5] Training and evaluating models...")
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Algorithm': name,
            'Accuracy (%)': round(acc * 100, 2),
            'Precision (%)': round(prec * 100, 2),
            'Recall (%)': round(rec * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
            'Time (s)': round(train_time, 2)
        })
        
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  F1-Score: {f1*100:.2f}%")
        print(f"  Training time: {train_time:.2f}s")
        
        # Save best model (Random Forest)
        if name == 'Random Forest':
            joblib.dump(model, 'models/model.pkl')
            print(f"\n  ✓ Saved Random Forest model to models/model.pkl")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
                    yticklabels=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        # Save with consistent naming
        filename = f'static/cm_{name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
    
    # Results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score (%)', ascending=False)
    
    print("\n[3/5] Final Results:")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)
    
    # Save results CSV
    results_df.to_csv('models/results.csv', index=False)
    
    # Create comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.2
    
    ax.bar(x - 1.5*width, results_df['Accuracy (%)'], width, label='Accuracy', color='#2E86AB')
    ax.bar(x - 0.5*width, results_df['Precision (%)'], width, label='Precision', color='#A23B72')
    ax.bar(x + 0.5*width, results_df['Recall (%)'], width, label='Recall', color='#F18F01')
    ax.bar(x + 1.5*width, results_df['F1-Score (%)'], width, label='F1-Score', color='#C73E1D')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison on NSL-KDD Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Algorithm'], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(80, 100)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/comparison_chart.png', dpi=150)
    plt.close()
    
    print("\n[4/5] Generated visualizations saved to /static/")
    print("[5/5] Training complete!")
    
    return results_df

if __name__ == '__main__':
    train_and_evaluate()