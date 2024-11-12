# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load Data (replace 'your_data.csv' with your actual data path)
data = pd.read_csv('your_data.csv')

# Select relevant features (replace with your chosen features)
features = ['previous_opponent_rank', 'minutes_played', 'shots_on_target', 'goal_attempts']
target = 'fantasy_points'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2)

# Create the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Evaluate model performance (e.g., Mean Squared Error)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, predictions, squared=False)
print('Root Mean Squared Error:', rmse)

# Predict points for new players (replace with player data)
new_player_data = pd.DataFrame({
    'previous_opponent_rank': [15],  # Replace with actual values
    'minutes_played': [80],
    'shots_on_target': [3],
    'goal_attempts': [5]
})

predicted_points = model.predict(new_player_data)[0]
print('Predicted Fantasy Points:', predicted_points)
