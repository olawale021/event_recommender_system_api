from flask import Flask, jsonify, request
import pickle
import pandas as pd
import logging
import os
from scipy.io import arff

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the hybrid model from the file
with open('hybrid_recommendation_model.pkl', 'rb') as file:
    hybrid_model = pickle.load(file)

svd_model = hybrid_model['svd_model']
knn_model = hybrid_model['knn_model']


# Load and check datasets
def load_data(filename, file_type='csv'):
    if file_type == 'csv':
        data = pd.read_csv(filename) if os.path.exists(filename) else pd.DataFrame()
    elif file_type == 'arff':
        data = pd.DataFrame(arff.loadarff(filename)[0]) if os.path.exists(filename) else pd.DataFrame()
    logger.debug(f"{filename} loaded with shape: {data.shape}")
    return data


# Load datasets
user_behavior = load_data('cleaned_user_behaviour_new.arff', file_type='arff')
events = load_data('events.csv')
preferred_categories = load_data('user_preferred_categories.csv')
tickets = load_data('tickets.csv')
users = load_data('users.csv')
friends = load_data('friends.csv')

# Decode object columns in user_behavior if loaded from arff
for col in user_behavior.select_dtypes([object]):
    user_behavior[col] = user_behavior[col].str.decode('utf-8')

# Map interaction values
interaction_values = {
    b'browse': 2,
    b'view': 3,
    b'purchase': 7,
    b'add_to_wishlist': 5,
    b'remove_from_wishlist': 0,
    b'view_friend_profile': 3
}
# Add other interactions if relevant for recommendations
relevant_interactions = {'purchase', 'add_to_wishlist'}


user_behavior['interaction_value'] = user_behavior['interaction_type'].map(interaction_values)


# Generate user relevant items
def get_user_relevant_items(user_behavior_df, relevant_interactions):
    relevant_df = user_behavior_df[user_behavior_df['interaction_type'].isin(relevant_interactions)]

    user_relevant_items = relevant_df.groupby('user_id')['event_id'].apply(set).to_dict()
    return user_relevant_items


user_relevant_items = get_user_relevant_items(user_behavior, relevant_interactions)


# Precision and Recall Calculation
def calculate_precision_recall(recommended_products, relevant_products):
    recommended_set = set(recommended_products)
    true_positives = recommended_set.intersection(relevant_products)
    precision = len(true_positives) / len(recommended_set) if recommended_set else 0
    recall = len(true_positives) / len(relevant_products) if relevant_products else 0
    return {"precision": precision, "recall": recall}


# Recommendation Functions
def get_hybrid_recommendations(user_id, events, user_behavior, preferred_categories, n_recommendations=5):
    purchased_events = user_behavior[(user_behavior['user_id'] == user_id) &
                                     (user_behavior['interaction_type'] == 'purchase')]['event_id'].unique()
    user_prefs = preferred_categories[preferred_categories['user_id'] == user_id]
    user_top_categories = set(user_prefs['category_id'].unique())

    preferred_category_boost = 0.1
    predictions = []
    for event_id in events['event_id'].unique():
        if event_id in purchased_events:
            continue
        try:
            svd_pred = svd_model.predict(user_id, event_id).est
            knn_pred = knn_model.predict(user_id, event_id).est
            hybrid_score = 0.6 * svd_pred + 0.4 * knn_pred
            event_category_id = events[events['event_id'] == event_id]['category_id'].values[0]
            if event_category_id in user_top_categories:
                hybrid_score += preferred_category_boost
            predictions.append({'event_id': event_id, 'hybrid_score': hybrid_score})
        except Exception:
            continue

    recommendations = pd.DataFrame(predictions).nlargest(n_recommendations, 'hybrid_score')

    # Merge with events data to include event details
    recommendations = recommendations.merge(events, on='event_id')
    return recommendations


def get_popular_events_anonymous(n=10):
    """Get popular events for anonymous users based on general popularity metrics"""
    try:
        # Calculate weighted interaction scores for events
        popular_events = (user_behavior
            .groupby('event_id')
            .agg({
                'interaction_value': 'sum',  # Total interaction value
                'user_id': 'count'          # Number of interactions
            })
            .reset_index()
            .assign(
                popularity_score=lambda df: (
                    df['interaction_value'] * 0.7 +  # Weight for interaction values
                    df['user_id'] * 0.3             # Weight for number of users
                )
            )
            .sort_values('popularity_score', ascending=False)
            .head(n))

        # Merge with event details
        recommendations = pd.merge(
            popular_events,
            events[['event_id', 'event_name', 'event_image', 'location', 'ticket_price']],
            on='event_id'
        )

        logger.debug(f"Found {len(recommendations)} popular events for anonymous users")
        return recommendations
    except Exception as e:
        logger.error(f"Error getting anonymous popular events: {str(e)}")
        return pd.DataFrame()


def recommend_popular_events(n=10, user_id=None):
    """Get popular event recommendations with support for both authenticated and anonymous users"""
    try:
        if user_id is None:
            # Anonymous user - return general popularity-based recommendations
            return get_popular_events_anonymous(n)

        # Authenticated user - return personalized popular recommendations
        popular_events = (user_behavior.groupby('event_id')['interaction_value']
                          .sum()
                          .reset_index()
                          .sort_values(by='interaction_value', ascending=False)
                          .head(n))

        recommendations = pd.merge(
            popular_events,
            events[['event_id', 'event_name', 'event_image', 'location', 'ticket_price']],
            on='event_id'
        )

        if user_id in user_relevant_items:
            relevant_items = user_relevant_items[user_id]
            precision_recall = calculate_precision_recall(recommendations['event_id'], relevant_items)
            recommendations['precision'] = precision_recall['precision']
            recommendations['recall'] = precision_recall['recall']

        return recommendations
    except Exception as e:
        logger.error(f"Error in recommend_popular_events for user_id {user_id}: {str(e)}")
        return pd.DataFrame()


def recommend_by_selected_category(user_id, n=10):
    user_categories = preferred_categories[preferred_categories['user_id'] == user_id]['category_id'].tolist()
    filtered_events = events[events['category_id'].isin(user_categories)]
    recommendations = filtered_events.sample(n=min(n, len(filtered_events)))
    if user_id in user_relevant_items:
        relevant_items = user_relevant_items[user_id]
        precision_recall = calculate_precision_recall(recommendations['event_id'], relevant_items)
        recommendations['precision'] = precision_recall['precision']
        recommendations['recall'] = precision_recall['recall']
    return recommendations


def profile_based_recommend(user_id, n=10):
    user_profile = user_behavior[user_behavior['user_id'] == user_id].copy()
    user_profile['weight'] = user_profile['interaction_value']
    user_interests = user_profile.groupby('event_id')['weight'].sum().reset_index()
    recommendations = pd.merge(user_interests, events[['event_id', 'event_name', 'event_image', 'location',
                                                       'ticket_price']], on='event_id')
    recommendations = recommendations.sort_values(by='weight', ascending=False).head(n)
    if user_id in user_relevant_items:
        relevant_items = user_relevant_items[user_id]
        precision_recall = calculate_precision_recall(recommendations['event_id'], relevant_items)
        recommendations['precision'] = precision_recall['precision']
        recommendations['recall'] = precision_recall['recall']
    return recommendations


# Recommendation Function: Friends-Based
def recommend_friends_events(user_id, n=10):
    user_friends = friends[friends['user_id'] == user_id]['friend_id'].tolist()
    logger.debug(f"User {user_id} friends: {user_friends}")

    friends_events = user_behavior[(user_behavior['user_id'].isin(user_friends)) &
                                   (user_behavior['interaction_type'].isin(relevant_interactions))]
    logger.debug(f"Friends' events for user {user_id}: {friends_events.shape[0]} interactions found.")

    if friends_events.empty:
        logger.debug(f"No relevant interactions found for friends of user {user_id}. Returning empty recommendations.")
        return pd.DataFrame()

    friends_event_scores = friends_events.groupby('event_id')['interaction_value'].sum().reset_index()
    recommendations = pd.merge(friends_event_scores, events[['event_id', 'event_name', 'event_image',
                                                             'location', 'ticket_price']],
                               on='event_id')
    recommendations = recommendations.sort_values(by='interaction_value', ascending=False).head(n)

    return recommendations[['event_id', 'event_name', 'location', 'ticket_price', 'event_image']]


# Flask Routes for Recommendations
@app.route('/recommend/hybrid', methods=['GET'])
def recommend_hybrid():
    user_id = int(request.args.get('user_id'))
    n = int(request.args.get('n', 10))
    recommendations = get_hybrid_recommendations(user_id, events, user_behavior, preferred_categories, n)
    return jsonify(recommendations.to_dict(orient='records'))


@app.route('/recommend/popular', methods=['GET'])
def recommend_popular():
    try:
        n = int(request.args.get('n', 5))
        user_id = request.args.get('user_id')

        # Handle both authenticated and anonymous users
        if user_id is not None:
            user_id = int(user_id)
            recommendations = recommend_popular_events(n=n, user_id=user_id)
        else:
            recommendations = recommend_popular_events(n=n, user_id=None)

        if recommendations.empty:
            logger.warning("No recommendations found")
            return jsonify([])

        return jsonify(recommendations.to_dict(orient='records'))

    except Exception as e:
        logger.error(f"Error processing popular recommendations request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/recommend/category', methods=['GET'])
def recommend_category():
    user_id = int(request.args.get('user_id'))
    n = int(request.args.get('n', 10))
    recommendations = recommend_by_selected_category(user_id=user_id, n=n)
    return jsonify(recommendations.to_dict(orient='records'))


@app.route('/recommend/profile', methods=['GET'])
def recommend_profile():
    user_id = int(request.args.get('user_id'))
    n = int(request.args.get('n', 10))
    recommendations = profile_based_recommend(user_id=user_id, n=n)
    return jsonify(recommendations.to_dict(orient='records'))


@app.route('/recommend/friends', methods=['GET'])
def recommend_friends():
    user_id = int(request.args.get('user_id'))
    n = int(request.args.get('n', 10))
    recommendations = recommend_friends_events(user_id=user_id, n=n)
    return jsonify(recommendations.to_dict(orient='records'))


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
