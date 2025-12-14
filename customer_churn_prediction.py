

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)




def load_and_clean_data(filepath):
    """
    Load the dataset and perform initial cleaning operations.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Cleaned dataframe
    """
    print("=" * 70)
    print("STEP 1: LOADING AND CLEANING DATA")
    print("=" * 70)
    
    # Load the data
    print("\n[1.1] Loading data...")
    df = pd.read_csv(filepath)
    print(f"    Original shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    
    
    print("\n[1.2] Converting 'TotalCharges' to numeric...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
   
    nan_count = df['TotalCharges'].isna().sum()
    print(f"    Found {nan_count} NaN values in 'TotalCharges'")
    
    
    if nan_count > 0:
        median_value = df['TotalCharges'].median()
        df['TotalCharges'].fillna(median_value, inplace=True)
        print(f"    Filled NaN values with median: {median_value:.2f}")
    
    # Drop customerID column
    print("\n[1.3] Dropping 'customerID' column...")
    df = df.drop('customerID', axis=1)
    print(f"    Shape after dropping customerID: {df.shape}")
    
    # Convert Churn to binary (Yes=1, No=0)
    print("\n[1.4] Converting 'Churn' to binary...")
    print(f"    Churn distribution before conversion:")
    print(df['Churn'].value_counts())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    print(f"    Churn distribution after conversion:")
    print(df['Churn'].value_counts())
    
    print("\n[OK] Data cleaning completed!")
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def identify_column_types(df, target_col='Churn'):
    """
    Automatically identify categorical and numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of the target column to exclude
        
    Returns:
    --------
    categorical_cols : list
        List of categorical column names
    numerical_cols : list
        List of numerical column names
    """
    # Exclude target column
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Identify categorical columns (object type or low cardinality integers)
    categorical_cols = []
    numerical_cols = []
    
    for col in feature_cols:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            # Check if it's actually categorical (low unique values)
            unique_vals = df[col].nunique()
            if unique_vals <= 10:  # Threshold for categorical
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    return categorical_cols, numerical_cols


def engineer_features(df, target_col='Churn'):
    """
    Perform feature engineering: one-hot encoding and scaling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of the target column
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    categorical_cols : list
        List of categorical column names
    numerical_cols : list
        List of numerical column names
    scaler : StandardScaler
        Fitted scaler object
    """
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    # Separate target and features
    y = df[target_col].copy()
    X = df.drop(target_col, axis=1).copy()
    
    # Identify column types
    print("\n[2.1] Identifying column types...")
    categorical_cols, numerical_cols = identify_column_types(df, target_col)
    
    print(f"    Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"    Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    
    # One-Hot Encoding for categorical variables
    print("\n[2.2] Applying One-Hot Encoding to categorical variables...")
    print(f"    Original shape: {X.shape}")
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    print(f"    Shape after encoding: {X_encoded.shape}")
    print(f"    Created {X_encoded.shape[1] - len(numerical_cols)} new binary features")
    
    # Scale numerical features
    print("\n[2.3] Scaling numerical features using StandardScaler...")
    scaler = StandardScaler()
    
    # Scale only the numerical columns that exist in the encoded dataframe
    numerical_cols_present = [col for col in numerical_cols if col in X_encoded.columns]
    
    if numerical_cols_present:
        X_encoded[numerical_cols_present] = scaler.fit_transform(X_encoded[numerical_cols_present])
        print(f"    Scaled {len(numerical_cols_present)} numerical features: {numerical_cols_present}")
    else:
        print("    No numerical features to scale")
    
    print("\n[OK] Feature engineering completed!")
    return X_encoded, y, categorical_cols, numerical_cols, scaler


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate their performance.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.array
        Training features
    X_test : pandas.DataFrame or numpy.array
        Test features
    y_train : pandas.Series or numpy.array
        Training target
    y_test : pandas.Series or numpy.array
        Test target
        
    Returns:
    --------
    models : dict
        Dictionary of trained models with their names
    results : dict
        Dictionary of model results
    """
    print("\n" + "=" * 70)
    print("STEP 3: MODEL TRAINING AND EVALUATION")
    print("=" * 70)
    
    models = {}
    results = {}
    
    # Define models
    model_configs = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    }
    
    # Train and evaluate each model
    for model_name, model in model_configs.items():
        print(f"\n[3.1] Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        models[model_name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        print(f"\n    Classification Report for {model_name}:")
        print("    " + "-" * 60)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        print(f"\n    Confusion Matrix for {model_name}:")
        print("    " + "-" * 60)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Store results
        results[model_name] = {
            'predictions': y_pred,
            'classification_report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'f1_score': report['1']['f1-score']  # F1 for churn class (1)
        }
    
    # Compare models
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':<15} {'F1-Score (Churn)':<20}")
    print("-" * 70)
    for model_name, result in results.items():
        print(f"{model_name:<25} {result['accuracy']:<15.4f} {result['f1_score']:<20.4f}")
    
    return models, results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_importance(models, X_encoded, top_n=15):
    """
    Plot feature importance from the best tree-based model.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_encoded : pandas.DataFrame
        Feature matrix with column names
    top_n : int
        Number of top features to display
    """
    print("\n" + "=" * 70)
    print("STEP 4: VISUALIZATION")
    print("=" * 70)
    
    # Get tree-based models
    tree_models = {}
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            tree_models[name] = model
    
    if not tree_models:
        print("No tree-based models found for feature importance plot.")
        return
    
    # Use the best performing tree-based model (XGBoost or Random Forest)
    best_model_name = None
    if 'XGBoost' in tree_models:
        best_model_name = 'XGBoost'
    elif 'Random Forest' in tree_models:
        best_model_name = 'Random Forest'
    else:
        best_model_name = list(tree_models.keys())[0]
    
    print(f"\n[4.1] Creating Feature Importance plot from {best_model_name}...")
    
    model = tree_models[best_model_name]
    feature_importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance - {best_model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("    [OK] Saved as 'feature_importance.png'")
    plt.show()


def plot_correlation_matrix(df, numerical_cols):
    """
    Plot correlation matrix of numerical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataframe
    numerical_cols : list
        List of numerical column names
    """
    print("\n[4.2] Creating Correlation Matrix plot...")
    
    # Filter numerical columns that exist in the dataframe
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    
    if not numerical_cols_present:
        print("    No numerical features found for correlation matrix.")
        return
    
    # Create correlation matrix
    corr_matrix = df[numerical_cols_present].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("    [OK] Saved as 'correlation_matrix.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to execute the complete pipeline.
    """
    # File path
    filepath = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Step 1: Data Cleaning
    df = load_and_clean_data(filepath)
    
    # Step 2: Feature Engineering
    X_encoded, y, categorical_cols, numerical_cols, scaler = engineer_features(df)
    
    # Step 3: Train-Test Split
    print("\n" + "=" * 70)
    print("STEP 3: DATA SPLITTING")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n    Training set: {X_train.shape[0]} samples")
    print(f"    Test set: {X_test.shape[0]} samples")
    print(f"    Churn rate in training set: {y_train.mean():.2%}")
    print(f"    Churn rate in test set: {y_test.mean():.2%}")
    
    # Step 4: Model Training and Evaluation
    models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Visualization
    plot_feature_importance(models, X_encoded, top_n=15)
    plot_correlation_matrix(df, numerical_cols)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - feature_importance.png")
    print("  - correlation_matrix.png")


if __name__ == "__main__":
    main()

