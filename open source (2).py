
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Load data
ratings = pd.read_csv('ratings.dat', sep='::', header=None, engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

# Create User x Item matrix
user_item_matrix = ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0).to_numpy()

# Save user ids for later use
user_ids = ratings['UserID'].unique()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(user_item_matrix)
labels = kmeans.labels_

# Group users by clusters
grouped_users = {i: np.where(labels == i)[0] for i in range(3)}

# Define the threshold for Approval Voting
THRESHOLD = 4

def additive_utilitarian(group):
    return np.sum(group, axis=0)

def average(group):
    return np.mean(group, axis=0)

def simple_count(group):
    return np.count_nonzero(group, axis=0)

def approval_voting(group):
    return np.sum(group >= THRESHOLD, axis=0)

def borda_count(group):
    ranks = group.argsort().argsort()
    return np.sum(ranks, axis=0)

def copeland_rule(group):
    comparisons = (group[:, :, None] > group[:, None, :]).sum(axis=0)
    return np.sum(comparisons - comparisons.T, axis=0)

# Create a dictionary of algorithms
algorithms = {
    'Additive Utilitarian': additive_utilitarian,
    'Average': average,
    'Simple Count': simple_count,
    'Approval Voting': approval_voting,
    'Borda Count': borda_count,
    'Copeland Rule': copeland_rule
}

# Function to get top 10 recommendations
def get_top_n_recommendations(scores, n=10):
    return np.argsort(scores)[-n:][::-1]

# Dictionary to store recommendations
recommendations = {}

for algo_name, algo_func in algorithms.items():
    recommendations[algo_name] = {}
    for group_id, users in grouped_users.items():
        group_matrix = user_item_matrix[users]
        scores = algo_func(group_matrix)
        top_recommendations = get_top_n_recommendations(scores)
        recommendations[algo_name][group_id] = top_recommendations

# Print recommendations
for algo_name, groups in recommendations.items():
    print(f"Algorithm: {algo_name}")
    for group_id, recs in groups.items():
        print(f"  Group {group_id}: {recs}")
