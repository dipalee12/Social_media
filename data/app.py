

# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load datasets
posts = pd.read_csv("posts_challenge.csv")
likes = pd.read_csv("likes_challenge.csv")
users = pd.read_csv("users_challenge.csv")

# Function to generate recommendations
def generate_recommendations(user_name, filter_option):
    # Assuming here that user_name is unique in the users dataset
    user_id = users[users['user_name'] == user_name]['user_id'].values[0]
    user_likes = likes[likes['user_id'] == user_id]['post_id'].tolist()
    user_posts = posts[posts['user_id'] == user_id]['post_id'].tolist()
    
    # Filter posts based on filter_option
    if filter_option == 'likes':
        filtered_posts = posts[~posts['post_id'].isin(user_likes)]
    elif filter_option == 'created':
        filtered_posts = posts[~posts['user_id'].isin([user_id])]
    
    # Calculate the number of likes for each post
    filtered_posts['likes_count'] = filtered_posts['post_id'].map(likes['post_id'].value_counts()).fillna(0)
    
    # Assign probability based on number of likes and recency
    filtered_posts['probability'] = 0.5  # Base probability
    filtered_posts.loc[filtered_posts['likes_count'] >= 5, 'probability'] += 0.3  # Increase probability for posts with 5 or more likes
    
    # Sort posts by probability and get top 10
    top_10_posts = filtered_posts.sort_values(by='probability', ascending=False).head(10)
    
    return top_10_posts[['post_id', 'probability']]

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for generating recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    user_name = request.form['user_name']
    filter_option = request.form['filter_option']
    rec_df = generate_recommendations(user_name, filter_option)
    return render_template('recommendations.html', recommendations=rec_df)

if __name__ == '__main__':
    app.run(debug=True)