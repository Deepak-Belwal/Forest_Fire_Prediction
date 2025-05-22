import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Random Forest import

# Load preprocessed dataset
df = pd.read_csv("D:/Projects/Forest Fire prediction/perfect_dataset.csv")

'''Create log1p_area for safety, this output variable is very skewed towards 0.0, thus it may make
    sense to model with the logarithm transform'''

df['log1p_area'] = np.log1p(df['area'])

# Remove top 1% extreme fires
threshold = df['area'].quantile(0.99)
df = df[df['area'] <= threshold]

# Define Features and Target
X = df.drop(columns=['area', 'log_area', 'X', 'Y','DC', 'wind', 'fire_occurred'])
y = df['log_area']

print(df.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,      
    max_depth=10,          
    min_samples_split=5,   
    min_samples_leaf=2,    
    max_features='sqrt',   
    random_state=42,
    n_jobs=-1             
)

#Fitting model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

#Making predictions on original area
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)

#Evaluating on original predictions
mae_original = mean_absolute_error(y_test_original, y_pred_original)
mse_original = mean_squared_error(y_test_original, y_pred_original)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


print(f"MAE on original scale: {mae_original:.2f}")
print(f"MSE on original scale: {mse_original:.2f}")

#Saving the model 
with open("brahmastra_2.pkl", "wb") as file1:
    pickle.dump(rf_model, file1)
print("Model saved as brahmastra_2 successfully!")

