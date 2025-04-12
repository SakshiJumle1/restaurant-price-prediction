# Restaurant Price Prediction with Restaurant Suggestions

# === 1. Import Libraries ===
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# === 2. Load Dataset ===
df = pd.read_csv("zomato.csv", encoding='latin-1')

# === 3. Data Preprocessing ===
df = df.drop_duplicates()
df = df[df['Aggregate rating'] > 0]
df = df[df['Average Cost for two'].notnull()]
df = df[['Restaurant Name', 'City', 'Cuisines', 'Has Online delivery', 'Has Table booking',
         'Aggregate rating', 'Average Cost for two']]
df = df.dropna()

le = LabelEncoder()
df['City'] = le.fit_transform(df['City'].astype(str))
df['Cuisines'] = le.fit_transform(df['Cuisines'].astype(str))
df['Has Online delivery'] = df['Has Online delivery'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Has Table booking'] = df['Has Table booking'].apply(lambda x: 1 if x == 'Yes' else 0)

X = df[['Aggregate rating', 'City', 'Cuisines', 'Has Online delivery', 'Has Table booking']]
y = df['Average Cost for two']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# === 4. Streamlit App ===
def run_app():
    st.title("Restaurant Price Prediction & Suggestions")

    rating = st.slider("Rating (0.0 - 5.0)", 0.0, 5.0, 3.5, 0.1)
    city = st.text_input("City")
    cuisines = st.text_input("Cuisine Type")
    online_order = st.selectbox("Online Order Available?", ["Yes", "No"])
    book_table = st.selectbox("Book Table Available?", ["Yes", "No"])

    if st.button("Predict Price and Show Restaurants"):
        city_enc = le.transform([city])[0] if city in le.classes_ else 0
        cuisine_enc = le.transform([cuisines])[0] if cuisines in le.classes_ else 0
        online_order_enc = 1 if online_order == "Yes" else 0
        book_table_enc = 1 if book_table == "Yes" else 0

        input_data = [[rating, city_enc, cuisine_enc, online_order_enc, book_table_enc]]
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted Price for Two: ₹{predicted_price:.2f}")

        # === 5. Suggest Similar Restaurants ===
        filtered_df = df[(df['Aggregate rating'] >= rating - 0.5) &
                         (df['Aggregate rating'] <= rating + 0.5) &
                         (df['Has Online delivery'] == online_order_enc) &
                         (df['Has Table booking'] == book_table_enc)]

        similar_restaurants = filtered_df.sort_values(by='Average Cost for two', key=lambda x: abs(x - predicted_price))

        st.subheader("Top Similar Restaurants")
        st.dataframe(similar_restaurants[['Restaurant Name', 'City', 'Cuisines', 'Aggregate rating', 'Average Cost for two']].head(10))

if __name__ == '__main__':
    run_app()
