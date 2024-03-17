import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
posts = pd.read_csv('posts_challenge.csv')
users = pd.read_csv('users_challenge.csv')
likes = pd.read_csv('likes_challenge.csv')
testing_users = pd.read_csv('testing_user.csv')

# Preprocess data
# Encode users and posts
user_encoder = LabelEncoder()
posts['user_id_encoded'] = user_encoder.fit_transform(posts['user_id'])
posts['post_id_encoded'] = user_encoder.fit_transform(posts['post_id'])

# Create user-item sparse matrix
def create_sparse_matrix(likes, num_users, num_posts):
    rows = user_encoder.transform(likes['user_id'])
    cols = user_encoder.transform(likes['post_id'])
    data = np.ones_like(rows)
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_posts))
    return user_item_matrix.astype(float)  # Ensure matrix is of floating-point data type

# Get the number of unique users and posts
num_users = len(user_encoder.classes_)
num_posts = posts['post_id_encoded'].nunique()

# Create user-item sparse matrix
user_item_matrix = create_sparse_matrix(likes, num_users, num_posts)

# SVD with sparse matrix
from scipy.sparse.linalg import svds

# Hyperparameters
embedding_dim = 50

# SVD decomposition
logger.info("Performing SVD decomposition...")
U, sigma, Vt = svds(user_item_matrix, k=embedding_dim)
logger.info("SVD decomposition completed.")

# Convert sigma to diagonal matrix
sigma = np.diag(sigma)

# Generate recommendations
def recommend_posts(user_id, n=10):
    user_idx = user_encoder.transform([user_id])[0]
    user_vector = np.dot(U[user_idx, :], sigma)
    scores = np.dot(user_vector, Vt)
    top_n_indices = np.argsort(scores)[::-1][:n]
    top_n_post_ids = user_encoder.inverse_transform(top_n_indices)
    return top_n_post_ids

# Format recommendations
logger.info("Generating recommendations for testing users...")
recommendations = pd.DataFrame(columns=['user_id', 'user_name', 'post_id'])
for index, user in testing_users.iterrows():
    recommended_posts = recommend_posts(user['user_id'])
    user_recommendations = pd.DataFrame({
        'user_id': [user['user_id']] * len(recommended_posts),
        'user_name': [user['user_name']] * len(recommended_posts),
        'post_id': recommended_posts
    })
    recommendations = pd.concat([recommendations, user_recommendations])
logger.info("Recommendations generated.")

# Save recommendations to CSV
logger.info("Saving recommendations to CSV...")
recommendations.to_csv('recommendations.csv', index=False)
logger.info("Recommendations saved to 'recommendations.csv'.")
