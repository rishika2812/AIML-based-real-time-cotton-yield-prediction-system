import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def remove_outliers_iqr(df):
    """Clean data by removing outliers using the Interquartile Range (IQR) method."""
    initial_shape = df.shape
    df_clean = df.copy()
    
    for col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Keep rows that are within the bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
    print(f"Removed {initial_shape[0] - df_clean.shape[0]} outliers. Rows remaining: {df_clean.shape[0]}")
    return df_clean

def build_dnn(input_dim):
    """Build a Deep Neural Network (128 -> 64 -> 32 -> 1 layers)."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    # 0. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, 'dataset', 'final_dataset.csv')
    models_dir = os.path.join(script_dir, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # 1. Load dataset
    print("1. Loading dataset...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Could not find {dataset_path}.")
        print("Please run 'python dataset/generate_dataset.py' to create the dataset first.")
        return
        
    # 2. Clean data and remove outliers using IQR
    print("2. Cleaning data and removing outliers using IQR...")
    df_cleaned = remove_outliers_iqr(df)
    
    # Separate features and target
    # Note: We drop 'Yield' to prevent data leakage because Yield is highly dependent on Biomass
    X = df_cleaned.drop(columns=['Biomass', 'Yield'])
    y = df_cleaned['Biomass']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Normalize features using MinMaxScaler
    print("3. Normalizing features using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler to models folder
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 4. Train Deep Neural Network
    print("4. Training Deep Neural Network (128->64->32->1)...")
    dnn_model = build_dnn(X_train_scaled.shape[1])
    
    # Train the model and keep track of history for plotting
    history = dnn_model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=150, batch_size=16, verbose=0
    )
    
    # Predict with DNN
    dnn_predictions = dnn_model.predict(X_test_scaled).flatten()
    
    # 5. Train Random Forest for comparison
    print("5. Training Random Forest model for comparison...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_predictions = rf_model.predict(X_test_scaled)
    
    # 6. Evaluate both using R² and RMSE
    print("6. Evaluating models...")
    
    # DNN Evaluation
    dnn_r2 = r2_score(y_test, dnn_predictions)
    dnn_rmse = np.sqrt(mean_squared_error(y_test, dnn_predictions))
    
    # RF Evaluation
    rf_r2 = r2_score(y_test, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    
    print("\n" + "="*40)
    print("     MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Deep Neural Network (DNN):")
    print(f"  - R² Score : {dnn_r2:.4f}")
    print(f"  - RMSE     : {dnn_rmse:.4f}")
    print(f"\nRandom Forest (RF):")
    print(f"  - R² Score : {rf_r2:.4f}")
    print(f"  - RMSE     : {rf_rmse:.4f}")
    print("="*40 + "\n")
    
    # 7. Save all models to models/ folder
    print("7. Saving models to models/ folder...")
    
    # Save DNN
    dnn_model_path = os.path.join(models_dir, 'dnn_model.keras')
    dnn_model.save(dnn_model_path)
    
    # Save RF
    rf_model_path = os.path.join(models_dir, 'rf_model.pkl')
    joblib.dump(rf_model, rf_model_path)
    
    print(f"Models successfully saved to {models_dir}/")
    
    # 8. Plot training curves and actual vs predicted graphs
    print("8. Plotting results...")
    plt.figure(figsize=(18, 5))
    sns.set_theme(style="whitegrid")
    
    # Plot A: DNN Training Curves (Loss)
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('DNN Training Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.legend()
    
    # Plot B: DNN Actual vs Predicted
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, dnn_predictions, alpha=0.6, color='blue')
    # Ideal prediction line
    min_val = min(y_test.min(), dnn_predictions.min())
    max_val = max(y_test.max(), dnn_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'DNN Actual vs Predicted\nR²: {dnn_r2:.3f} | RMSE: {dnn_rmse:.2f}')
    plt.xlabel('Actual Biomass (kg/ha)')
    plt.ylabel('Predicted Biomass (kg/ha)')
    plt.legend()
    
    # Plot C: RF Actual vs Predicted
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, rf_predictions, alpha=0.6, color='green')
    # Ideal prediction line
    min_val_rf = min(y_test.min(), rf_predictions.min())
    max_val_rf = max(y_test.max(), rf_predictions.max())
    plt.plot([min_val_rf, max_val_rf], [min_val_rf, max_val_rf], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'Random Forest Actual vs Predicted\nR²: {rf_r2:.3f} | RMSE: {rf_rmse:.2f}')
    plt.xlabel('Actual Biomass (kg/ha)')
    plt.ylabel('Predicted Biomass (kg/ha)')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plots_path = os.path.join(models_dir, 'evaluation_plots.png')
    plt.savefig(plots_path)
    print(f"Plots successfully saved to {plots_path}")
    print("Opening plots window...")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
