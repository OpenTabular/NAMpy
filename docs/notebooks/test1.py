import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Import NAMpy models
from nampy.models import NAMRegressor, NBMRegressor, GPNAMRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic regression data
X, y = make_regression(
    n_samples=10000, 
    n_features=5, 
    n_informative=5,
    noise=10.0, 
    random_state=42
)

# Convert to DataFrame for better visualization
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

print(f"Dataset shape: {X_df.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
X_df.head()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create and train the NAM model
model = NAMRegressor(
    numerical_preprocessing="standardization",  # Preprocessing method
    dropout=0.1,
    layer_sizes=[64, 32, 16],  # Hidden layer sizes for each feature network
)

# Fit the model
model.fit(
    X_train, 
    y_train, 
    max_epochs=20,
    lr=1e-3,
    patience=10,  # Early stopping patience
    batch_size=128
)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

# Visualize predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'NAM Regression: Predicted vs Actual (R² = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.show()

model.plot(X_test, y_test)