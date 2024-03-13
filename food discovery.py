import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tkinter import *


# Function to convert 'rate' column to numeric type
def convert_to_numeric_rate(rate):
    try:
        return float(rate)
    except ValueError:
        return 0.0  # Return 0.0 if conversion fails


# Function to recommend restaurants based on user preferences using KNN
def recommend_restaurants():
    # Example user preferences
    user_preferences = {}
    user_preferences['location'] = location_entry.get()
    user_preferences['rest_type'] = rest_type_entry.get()
    user_preferences['online_order'] = online_order_var.get()
    user_preferences['book_table'] = book_table_var.get()
    user_preferences['rate'] = float(rate_entry.get())

    # Convert 'rate' column to numeric type
    restaurant_data['rate'] = restaurant_data['rate'].apply(convert_to_numeric_rate)

    # Filter restaurants based on user preferences
    filtered_restaurants = restaurant_data[(restaurant_data['location'] == user_preferences['location']) &
                                           (restaurant_data['rest_type'] == user_preferences['rest_type']) &
                                           (restaurant_data['online_order'] == user_preferences['online_order']) &
                                           (restaurant_data['book_table'] == user_preferences['book_table']) &
                                           (restaurant_data['rate'] >= user_preferences['rate'])]

    if filtered_restaurants.empty:
        recommendation_label.config(text="No restaurants found matching the provided preferences.")
        return

    # Select relevant features for recommendation
    features = ['rate', 'location', 'rest_type', 'online_order', 'book_table']
    X = filtered_restaurants[features]

    # Preprocessing for numerical features
    numeric_features = ['rate']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # Preprocessing for categorical features
    categorical_features = ['location', 'rest_type', 'online_order', 'book_table']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', 'passthrough', categorical_features)])

    # Define KNN model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine'))
    ])

    # Fit the model
    model.fit(X)

    # Transform user preferences
    user_preferences_transformed = preprocessor.transform(pd.DataFrame([user_preferences]))

    # Find nearest neighbors
    distances, indices = model.kneighbors(user_preferences_transformed)

    # Return recommended restaurants
    recommended_restaurants = filtered_restaurants.iloc[indices[0]]
    recommendation_label.config(text=recommended_restaurants.head(3).to_string(index=False))


# Load restaurant data from CSV file
restaurant_data = pd.read_csv('zomato.csv')

# Create GUI
root = Tk()
root.title("Restaurant Recommender")

# Location
location_label = Label(root, text="Location:")
location_label.grid(row=0, column=0, padx=5, pady=5, sticky=W)
location_entry = Entry(root)
location_entry.grid(row=0, column=1, padx=5, pady=5)

# Restaurant Type
rest_type_label = Label(root, text="Restaurant Type:")
rest_type_label.grid(row=1, column=0, padx=5, pady=5, sticky=W)
rest_type_entry = Entry(root)
rest_type_entry.grid(row=1, column=1, padx=5, pady=5)

# Online Order
online_order_label = Label(root, text="Online Order:")
online_order_label.grid(row=2, column=0, padx=5, pady=5, sticky=W)
online_order_var = StringVar(value="Yes")
online_order_entry = OptionMenu(root, online_order_var, "Yes", "No")
online_order_entry.grid(row=2, column=1, padx=5, pady=5)

# Book Table
book_table_label = Label(root, text="Book Table:")
book_table_label.grid(row=3, column=0, padx=5, pady=5, sticky=W)
book_table_var = StringVar(value="Yes")
book_table_entry = OptionMenu(root, book_table_var, "Yes", "No")
book_table_entry.grid(row=3, column=1, padx=5, pady=5)

# Rate
rate_label = Label(root, text="Minimum Rating:")
rate_label.grid(row=4, column=0, padx=5, pady=5, sticky=W)
rate_entry = Entry(root)
rate_entry.grid(row=4, column=1, padx=5, pady=5)

# Recommend Button
recommend_button = Button(root, text="Recommend Restaurants", command=recommend_restaurants)
recommend_button.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

# Recommended Restaurants Label
recommendation_label = Label(root, text="")
recommendation_label.grid(row=6, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()
