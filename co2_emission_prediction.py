# ==========================
# 🌍 CO₂ Emission Predictor – SDG 13: Climate Action
# Author: Your Name
# Description:
# Predicts carbon emissions based on GDP and energy consumption
# to support climate action and sustainable development.
# ==========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "Country": ["Kenya", "Uganda", "Tanzania", "Nigeria", "South Africa", "Egypt", "Ethiopia", "Ghana", "Morocco", "Algeria"],
    "GDP": [95, 45, 70, 450, 400, 500, 120, 65, 135, 160],
    "Energy_Consumption": [25, 15, 22, 180, 200, 210, 40, 35, 60, 80],
    "CO2_Emissions": [18, 10, 14, 130, 150, 160, 25, 20, 40, 50]
}
df = pd.DataFrame(data)

X = df[['GDP', 'Energy_Consumption']]
y = df['CO2_Emissions']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")

plt.figure(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, color='green', s=80)
plt.xlabel("Actual CO₂ Emissions (Mt)")
plt.ylabel("Predicted CO₂ Emissions (Mt)")
plt.title("CO₂ Emission Prediction Results")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.show()

new_data = pd.DataFrame({
    "GDP": [200],
    "Energy_Consumption": [120]
})
predicted_emission = model.predict(new_data)[0]
print(f"\nPredicted CO₂ Emission for GDP=200B USD and Energy=120TWh: {predicted_emission:.2f} Mt")
