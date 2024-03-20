import pandas as pd
import numpy as np
from copy import deepcopy
from geopy.distance import geodesic

# Load the dataset
input_file_path = '../../input_datasets/part_a/part_a_input_dataset_4.csv'
data = pd.read_csv(input_file_path)

# Display the first few rows to understand the structure
data.head()
# Initialize the locations including the depot
# First, extract the depot coordinates, which are assumed to be the same for each row
depot_coordinates = (data.iloc[0]['depot_lat'], data.iloc[0]['depot_lng'])

# Add depot as the first location and then append all other locations
locations = [depot_coordinates] + list(data[['lat', 'lng']].apply(tuple, axis=1))

# Reimplement the PSO using the geodesic library for accurate distance calculations
class Particle:
    def __init__(self, sequence):
        self.position = sequence
        self.best_position = sequence
        self.best_distance = float('inf')
        self.velocity = [0] * (len(sequence) - 2)  # Exclude the depot from velocity considerations

# Initialize PSO parameters
n_particles = 500
iterations = 200

# Initialize the swarm
swarm = []
for _ in range(n_particles):
    sequence = np.arange(1, len(locations))
    np.random.shuffle(sequence)
    sequence = np.insert(sequence, 0, 0)  # Start from the depot
    sequence = np.append(sequence, 0)  # End at the depot
    swarm.append(Particle(sequence))

# Define the fitness function using geodesic distance
def calculate_geodesic_distance(sequence, locations):
    total_distance = 0
    for i in range(len(sequence) - 1):
        total_distance += geodesic(locations[sequence[i]], locations[sequence[i + 1]]).kilometers
    return total_distance

# PSO loop
global_best_distance = float('inf')
global_best_position = None

for _ in range(iterations):
    for particle in swarm:
        current_distance = calculate_geodesic_distance(particle.position, locations)
        
        # Update personal and global bests
        if current_distance < particle.best_distance:
            particle.best_position = deepcopy(particle.position)
            particle.best_distance = current_distance
        if current_distance < global_best_distance:
            global_best_distance = current_distance
            global_best_position = deepcopy(particle.position)
        
        # Velocity and position updates (simplified version with swap operations)
        for i in range(1, len(particle.position) - 1):
            if np.random.rand() < 0.5:  # Swap probability
                j = np.random.randint(1, len(particle.position) - 1)
                particle.position[i], particle.position[j] = particle.position[j], particle.position[i]

# Prepare the output DataFrame based on the best route
best_route_indices = global_best_position[1:-1]  # Exclude depot from the sequence
output_data = data.iloc[best_route_indices - 1].copy()  # Adjust indices for zero-based indexing
output_data['dlvr_seq_num'] = range(1, len(best_route_indices) + 1)

# Define the output file path
output_file_path = '../../output_datasets/part_a/part_a_dataset_4.csv'

# Save the output DataFrame to a CSV file
output_data.to_csv(output_file_path, index=False)

# Return the path of the generated output file
output_file_path
# Calculate and return the total distance for the optimized route
total_optimized_distance = calculate_geodesic_distance(global_best_position, locations)
print(total_optimized_distance)
data1 = {
    "Dataset": ["part_a_input_dataset_4"],
    "Best Route Distance": [total_optimized_distance]
}
df = pd.DataFrame(data1)

# Load existing CSV file if it exists, otherwise create a new DataFrame
try:
    existing_df = pd.read_csv("../../output_datasets/part_a_best_routes_distance_travelled.csv")
except FileNotFoundError:
    existing_df = pd.DataFrame(columns=["Dataset", "Best Route Distance"])

# Append the new data to the existing DataFrame
df_to_append = pd.concat([existing_df, df], ignore_index=True)

# Write the updated DataFrame back to the CSV file
df_to_append.to_csv("../../output_datasets/part_a_best_routes_distance_travelled.csv", index=False)