import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


file_path = 'D:\Project Task\Task5\Singapore resale flat prices\Resaleflatprices.csv'
data = pd.read_csv(file_path)

data = pd.get_dummies(data, columns=['town'], drop_first=True)

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

features = data[['month','flat_type', 'block', 'street_name', 'storey_range','floor_area_sqm','flat_model','lease_commence_date','remaining_lease','resale_price']]

current_year = 2023
features['flat_age'] = current_year - data['lease_commence_date']

X = features
y = data['resale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

st.title("Singapore Resale Flat Price Predictor")

town = st.selectbox("Select Town", data['town'].unique())
flat_type = st.selectbox("Select Flat Type", data['flat_type'].unique())

if st.button("Predict Resale Price"):
    user_inputs = pd.DataFrame({'town': [town], 'flat_type': [flat_type]})  # Add other user inputs
    prediction = model.predict(user_inputs)
    st.success(f"Predicted Resale Price: {prediction[0]:,.2f} SGD")
