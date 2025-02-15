def get_building_edges(slice):
    # ... existing edge detection code ...
    
    def edge_density(edge, all_edges, radius=10):
        # Count how many other edges are nearby
        edge_points = np.vstack(all_edges)
        density = 0
        for point in edge:
            distances = np.sqrt(np.sum((edge_points - point)**2, axis=1))
            density += np.sum(distances < radius)
        return density / len(edge)
    
    # Filter edges by density
    max_density = 5.0  # Adjust this threshold
    filtered_edges = [edge for edge in edges if edge_density(edge, edges) < max_density]
    
    return np.array(filtered_edges) 