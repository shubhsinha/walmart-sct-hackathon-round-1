import pandas as pd
import numpy as np
from copy import deepcopy
from geopy.distance import geodesic

# Load the dataset from the provided CSV file
input_file_path = '../../input_datasets/part_b/part_b_input_dataset_1.csv'
data = pd.read_csv(input_file_path)

# Display the first few rows of the dataset to understand its structure
data.head()
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import haversine_distances

# Define a function to calculate the distance matrix using the Haversine formula
def calculate_distance_matrix(locations):
    # Convert lat/long from degrees to radians
    locations_rad = np.radians(locations)

    # Use the haversine formula to calculate the distance matrix
    dist_matrix = haversine_distances(locations_rad) * 6371  # Earth radius in kilometers

    return dist_matrix

# Prepare the location data (including the depot as the first point)
locations = data[['lat', 'lng']].values
depot_location = np.array([[data['depot_lat'].iloc[0], data['depot_lng'].iloc[0]]])
all_locations = np.vstack((depot_location, locations))

# Calculate the distance matrix
distance_matrix = calculate_distance_matrix(all_locations)

# Check the shape of the distance matrix (should be n+1 x n+1, where n is the number of orders)
distance_matrix.shape
import numpy as np

# Define a simplified Particle Swarm Optimization (PSO) implementation
class ParticleSwarmOptimizer:
    def __init__(self, distance_matrix, num_particles=30, max_iter=100):
        self.distance_matrix = distance_matrix
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.num_locations = distance_matrix.shape[0] - 1  # Excluding the depot

        # PSO parameters
        self.w = 0.729  # Inertia weight
        self.c1 = 1.49445  # Cognitive constant
        self.c2 = 1.49445  # Social constant

        # Initialize particles
        self.particles = [np.random.permutation(self.num_locations) for _ in range(num_particles)]
        self.velocities = [np.zeros(self.num_locations) for _ in range(num_particles)]

        # Initialize personal bests and global best
        self.pbests = self.particles.copy()
        self.pbest_distances = [self.calculate_route_distance(p) for p in self.pbests]
        self.gbest = self.pbests[np.argmin(self.pbest_distances)]
        self.gbest_distance = min(self.pbest_distances)

    def calculate_route_distance(self, sequence):
        # Calculate the total distance of the route
        total_distance = self.distance_matrix[0, sequence[0] + 1]
        for i in range(1, len(sequence)):
            total_distance += self.distance_matrix[sequence[i - 1] + 1, sequence[i] + 1]
        total_distance += self.distance_matrix[sequence[-1] + 1, 0]
        return total_distance

    def update_velocity(self, particle, velocity, pbest, gbest):
        # Update the velocity based on current velocity, personal best, and global best
        r1, r2 = np.random.rand(2, self.num_locations)
        new_velocity = (self.w * velocity +
                        self.c1 * r1 * (pbest - particle) +
                        self.c2 * r2 * (gbest - particle))
        return new_velocity

    def update_particle(self, particle, velocity):
        # Apply the velocity updates to the particle (position)
        new_particle = particle + velocity
        new_particle = np.argsort(new_particle)
        return new_particle

    def optimize(self):
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocities and particles
                self.velocities[i] = self.update_velocity(self.particles[i], self.velocities[i], self.pbests[i], self.gbest)
                self.particles[i] = self.update_particle(self.particles[i], self.velocities[i])

                # Update personal best
                current_distance = self.calculate_route_distance(self.particles[i])
                if current_distance < self.pbest_distances[i]:
                    self.pbests[i] = self.particles[i]
                    self.pbest_distances[i] = current_distance

                    # Update global best
                    if current_distance < self.gbest_distance:
                        self.gbest = self.particles[i]
                        self.gbest_distance = current_distance

        return self.gbest, self.gbest_distance

# Initialize and run the optimizer
optimizer = ParticleSwarmOptimizer(distance_matrix, num_particles=50, max_iter=200)
best_route, best_distance = optimizer.optimize()

# Adjust indices and add depot
optimized_route_indices = np.concatenate(([0], best_route + 1, [0]))

# Create the output DataFrame based on the optimized route
output_data = data.iloc[best_route].copy()
output_data['dlvr_seq_num'] = range(1, len(best_route) + 1)

# Add depot data at the beginning and end
output_data = pd.concat([
    pd.DataFrame({'order_id': [0], 'lng': [data['depot_lng'].iloc[0]], 'lat': [data['depot_lat'].iloc[0]],
                  'depot_lat': [data['depot_lat'].iloc[0]], 'depot_lng': [data['depot_lng'].iloc[0]], 'dlvr_seq_num': [0]}),
    output_data,
    pd.DataFrame({'order_id': [0], 'lng': [data['depot_lng'].iloc[0]], 'lat': [data['depot_lat'].iloc[0]],
                  'depot_lat': [data['depot_lat'].iloc[0]], 'depot_lng': [data['depot_lng'].iloc[0]], 'dlvr_seq_num': [len(best_route) + 1]})
], ignore_index=True)

# Save to CSV
output_file_path = '../../output_datasets/part_b/part_b_dataset_1.csv'
output_data.to_csv(output_file_path, index=False)

output_file_path
def calculate_geodesic_distance(sequence, locations):
    total_distance = 0
    for i in range(len(sequence) - 1):
        total_distance += geodesic(locations[sequence[i]], locations[sequence[i + 1]]).kilometers
    return total_distance
total_optimized_distance = calculate_geodesic_distance(global_best_position, locations)
print(total_optimized_distance)
data1 = {
    "Dataset": ["part_a_input_dataset_1"],
    "Best Route Distance": [total_optimized_distance]
}
df = pd.DataFrame(data1)

# Load existing CSV file if it exists, otherwise create a new DataFrame
try:
    existing_df = pd.read_csv("../../output_datasets/part_b_best_routes_distance_travelled.csv")
except FileNotFoundError:
    existing_df = pd.DataFrame(columns=["Dataset", "Best Route Distance"])

# Append the new data to the existing DataFrame
df_to_append = pd.concat([existing_df, df], ignore_index=True)

# Write the updated DataFrame back to the CSV file
df_to_append.to_csv("../../output_datasets/part_b_best_routes_distance_travelled.csv", index=False)