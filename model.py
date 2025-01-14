import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Train and save the model (you can also run this separately as needed)
def train_model():
    data = pd.DataFrame({
        'Size': [1500, 1800, 2400, 3000, 3500],  # Size of the house (sq ft)
        'Bedrooms': [3, 4, 3, 5, 4],  # Number of bedrooms
        'Price': [400000, 500000, 600000, 650000, 700000]  # House prices ($)
    })

    X = data[['Size', 'Bedrooms']]
    y = data['Price']

    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model as a .pkl file
    joblib.dump(model, 'house_price_model.pkl')
    print("Model trained and saved as 'house_price_model.pkl'.")

# Load the trained model
def load_model():
    return joblib.load('house_price_model.pkl')
