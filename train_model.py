import pandas as pd
from scipy.io import arff
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

def train_and_save_model():
    # Load the original dataset with interaction types
    file_path = 'cleaned_user_behaviour.arff'
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)

    # Load additional datasets
    events_df = pd.read_csv('events.csv')
    users_df = pd.read_csv('users.csv')
    preferred_categories_df = pd.read_csv('user_preferred_categories.csv')

    # Define interaction values for different types of interactions
    interaction_values = {
        b'browse': 2,
        b'view': 3,
        b'purchase': 4,
        b'add_to_wishlist': 5,
        b'remove_from_wishlist': -1,
        b'view_friend_profile': 1
    }

    # Map interaction types to their values
    df['interaction_value'] = df['interaction_type'].map(interaction_values)

    # Create the user-item interaction matrix from all interactions
    interactions = df[['user_id', 'event_id', 'interaction_value']]
    interactions = interactions.dropna(subset=['interaction_value'])

    # Define the Reader object for Surprise
    reader = Reader(rating_scale=(-1, 5))

    # Load the dataset into Surprise
    data = Dataset.load_from_df(interactions[['user_id', 'event_id', 'interaction_value']], reader)

    # Split the dataset into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Create an instance of the SVD algorithm
    model = SVD()
    model.fit(trainset)

    # Save the trained model to a file
    joblib.dump(model, 'svd_model.pkl')
    print("Model saved as 'svd_model.pkl'")

    # Evaluate the model on the test set
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    train_and_save_model()
