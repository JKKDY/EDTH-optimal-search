from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from scipy import ndimage




COLORS = {
    'field': ['#edf0d4', '#cdebb0', '#87e0be', "#ecedd5", "#e6e8d0", "#f4f5dd", "#e3e6ce", "#cee5b1", "#f1eee8"],
    'trees': ['#add19d', '#c8d7aa', "#c8d3ad", "#b4d0a2", "#a0bb8e", "#a4c896", "#a0c491", "#bcdab1", "#c1dab5", "#c1d9b6"],
    'street': ['#ffffff', '#f6fabf', '#a6822b', "#717171", "#dddde8", "#f6fabf", "#ac8332", "#f9fac7"],
    'other': ['#aedfa2', '#87e0be', '#f2dad9', "#ead6b8"],
    'building': ['#e0dfdf', '#f5dcba', "#d9d0c9", "#c3b5ab", "#d9d9d9", "#dfdfdf", "#f2eee8", "#d5cdc7", "#fdfde5"],
    'water': ['#abd3de', '#cdebb0']
}

tolerance = 15

def read_image(file):
    return Image.open(file)

def check_color_present(slice_img, colors, tolerance=tolerance):
    return any(check_single_color(slice_img, color, tolerance) for color in colors)

def check_single_color(slice_img, target_color, tolerance=15):
    np_slice = np.array(slice_img)
    # Convert hex to RGB
    r = int(target_color[1:3], 16)
    g = int(target_color[3:5], 16)
    b = int(target_color[5:7], 16)
    
    # Check ALL pixels in the slice, returns True if ANY pixel matches
    return np.any(
        (np.abs(np_slice[:,:,0] - r) < tolerance) &  # R channel
        (np.abs(np_slice[:,:,1] - g) < tolerance) &  # G channel
        (np.abs(np_slice[:,:,2] - b) < tolerance)    # B channel
    )



def calculate_normals(edge_points, np_slice, tree_colors, tolerance=10):
    """Calculate normals for edge points with color checking."""
    normals = np.full((len(edge_points), 2), np.nan)  # Using NaN for undefined normals
    offset = 5  # Using 5 indices ahead for tangent calculation

    def is_tree_color(pixel, tol=tolerance+5):
        # Assume tree_colors are strings like "#AABBCC"
        for tree_color in tree_colors:
            r = int(tree_color[1:3], 16)
            g = int(tree_color[3:5], 16)
            b = int(tree_color[5:7], 16)
            if (abs(int(pixel[0]) - r) < tol and 
                abs(int(pixel[1]) - g) < tol and 
                abs(int(pixel[2]) - b) < tol):
                return True
        return False

    # Loop over edge points where we can safely look ahead by "offset" indices
    for i in range(len(edge_points) - offset):
        x, y = edge_points[i]
        # Compute tangent using the point "offset" steps ahead
        dx = edge_points[i+offset][0] - x
        dy = edge_points[i+offset][1] - y

        length = np.sqrt(dx*dx + dy*dy)
        if length == 0:
            continue  # Skip degenerate cases

        # Compute a normal; note: here, you use (dy, dx) which is a -90° rotation compared to (-dy, dx)
        # Confirm that this is the rotation you want.
        normal_x = dy / length
        normal_y = dx / length

        # Set up check distances in pixel units.
        # Adjust these values if needed. For example, if your image is in pixels, distances like 1 or 2 might be more appropriate.
        check_dists = [5, 8, 10, 13, 15]  
        pos_tree_count = 0
        neg_tree_count = 0
        valid_checks = 0

        for dist in check_dists:
            # Use round() to pick the nearest pixel
            pos_x = int(round(x + normal_x * dist))
            pos_y = int(round(y - normal_y * dist))
            neg_x = int(round(x - normal_x * dist))
            neg_y = int(round(y + normal_y * dist))

            # Ensure the indices are within bounds of the image
            if (0 <= pos_y < np_slice.shape[0] and 
                0 <= pos_x < np_slice.shape[1] and
                0 <= neg_y < np_slice.shape[0] and
                0 <= neg_x < np_slice.shape[1]):

                if is_tree_color(np_slice[pos_y, pos_x]):
                    valid_checks += 1
                    pos_tree_count += 1
                if is_tree_color(np_slice[neg_y, neg_x]):
                    valid_checks += 1
                    neg_tree_count += 1

        # Only set the normal if you have sufficient color evidence.
        if valid_checks > 4:
            # Decide which side has more tree color
            if pos_tree_count > neg_tree_count:  # More tree pixels on the positive side
                normals[i] = [-normal_x, -normal_y]  # Reverse the normal
            else:
                normals[i] = [normal_x, normal_y]
        else:
            normals[i] = [np.nan, np.nan]  # Or leave it undefined

    return normals



def get_tree_edges(slice_img, tolerance=tolerance+5):
    np_slice = np.array(slice_img)
    combined_mask = np.zeros(np_slice.shape[:2], dtype=bool)
    
    # Combine masks for all tree colors
    for tree_color in COLORS['trees']:
        r = int(tree_color[1:3], 16)
        g = int(tree_color[3:5], 16)
        b = int(tree_color[5:7], 16)
        
        tree_mask = (
            (np.abs(np_slice[:,:,0] - r) < tolerance) &
            (np.abs(np_slice[:,:,1] - g) < tolerance) &
            (np.abs(np_slice[:,:,2] - b) < tolerance)
        )
        combined_mask = combined_mask | tree_mask
    
    # Use Canny edge detection with higher threshold
    edges = feature.canny(combined_mask.astype(float), 
                         sigma=4,
                         low_threshold=0.3,
                         high_threshold=0.6)
    
    edge_coords = np.where(edges)
    if len(edge_coords[0]) == 0:
        return np.array([]), np.array([])
    
    # Convert to x,y coordinates
    edge_points = np.column_stack((edge_coords[1], edge_coords[0]))
    
    # Now we only get back the high-confidence points and their normals
    normals = calculate_normals(edge_points, np_slice, COLORS['trees'], tolerance=tolerance)
    
    return edge_points, normals

def get_building_edges(slice_img, tolerance=tolerance):
    np_slice = np.array(slice_img)
    combined_mask = np.zeros(np_slice.shape[:2], dtype=bool)

    # Combine masks for all tree colors
    for building_color in COLORS['building']:
        r = int(building_color[1:3], 16)
        g = int(building_color[3:5], 16)
        b = int(building_color[5:7], 16)
        
        building_mask = (
            (np.abs(np_slice[:,:,0] - r) < tolerance) &
            (np.abs(np_slice[:,:,1] - g) < tolerance) &
            (np.abs(np_slice[:,:,2] - b) < tolerance)
        )
        combined_mask = combined_mask | building_mask

    # Use Canny edge detection with higher threshold
    edges = feature.canny(combined_mask.astype(float), 
                         sigma=4,
                         low_threshold=0.3,
                         high_threshold=0.6)
    
    edge_coords = np.where(edges)

    if len(edge_coords[0]) > 0:
        edge_points = np.column_stack((edge_coords[1], edge_coords[0]))
        # Now we only get back the high-confidence points and their normals
        normals = calculate_normals(edge_points, np_slice, COLORS['building'], tolerance=tolerance)
        return edge_points, normals
    else:
        return np.array([]), np.array([])


def generate_direction_vector(contains, tree_normals, building_normals):
    """
    Generate 8D vector
    """

    #contains: field, trees, street, other, building, water
    #if only field, all directions are super easy to detect
    if (contains[0] or contains[2] or contains[5]) and not contains[1] and not contains[3] and not contains[4]:
        return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    else:
        #trees
        if contains[1]:
            #check tree normal validity
            #extract only the vectors that are not nan
            if tree_normals.shape[0] > 0:
                valid_normals = tree_normals[~np.isnan(tree_normals).any(axis=1)]
                average_normal = np.array([np.mean(valid_normals[:,0]), np.mean(valid_normals[:,1])])
                arr = np.array([0.5, 0.1, 0.5, 0.9, 0.5, 0.3, 0.5, 0.9])
                if average_normal[0] > 0:
                    if average_normal[1] > 0:
                        #showing from middle to sector 1
                        arr[[0, 1]] = arr[[1, 0]]
                        arr[[2, 3]] = arr [[3, 2]]
                        arr[[4, 5]] = arr [[5,4]]
                        arr[[6, 7]] = arr [[7, 6]]
                        return arr #change index 2 to 4 and 6 and 8
                    else:
                        #showing from middle to sector 4
                        arr[[1, 3]] = arr[[3, 1]]
                        arr[[0, 2]] = arr [[2, 0]]
                        arr[[5, 7]] = arr [[7,5]]
                        arr[[4, 6]] = arr [[6, 4]]
                        return arr
                else:
                    if average_normal[1] > 0:
                        #showing from middle to sector 2
                        return arr
                    else:
                        #showing from middle to sector 3
                        arr[[1, 2]] = arr[[2, 1]]
                        arr[[0, 3]] = arr [[3, 0]]
                        arr[[5, 6]] = arr [[6,5]]
                        arr[[4, 7]] = arr [[7, 4]]
                        return arr 
            else:
                return np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        if contains[4]:
            #check tree normal validity
            #extract only the vectors that are not nan
            if building_normals.shape[0] > 0:
                valid_normals = building_normals[~np.isnan(building_normals).any(axis=1)]
                average_normal = np.array([np.mean(valid_normals[:,0]), np.mean(valid_normals[:,1])])
                arr = np.array([0.5, 0.1, 0.5, 0.9, 0.5, 0.3, 0.5, 0.9])
                if average_normal[0] > 0:
                    if average_normal[1] > 0:
                        #showing from middle to sector 1
                        arr[[0, 1]] = arr[[1, 0]]
                        arr[[2, 3]] = arr [[3, 2]]
                        arr[[4, 5]] = arr [[5,4]]
                        arr[[6, 7]] = arr [[7, 6]]
                        return arr #change index 2 to 4 and 6 and 8
                    else:
                        #showing from middle to sector 4
                        arr[[1, 3]] = arr[[3, 1]]
                        arr[[0, 2]] = arr [[2, 0]]
                        arr[[5, 7]] = arr [[7,5]]
                        arr[[4, 6]] = arr [[6, 4]]
                        return arr
                else:
                    if average_normal[1] > 0:
                        #showing from middle to sector 2
                        return arr
                    else:
                        #showing from middle to sector 3
                        arr[[1, 2]] = arr[[2, 1]]
                        arr[[0, 3]] = arr [[3, 0]]
                        arr[[5, 6]] = arr [[6,5]]
                        arr[[4, 7]] = arr [[7, 4]]
                        return arr 
            else:
                return np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
              
    return np.array([1., 1., 1., 1., 1., 1., 1., 1.])


def slice_image(image, n_rows, n_cols):
    width, height = image.size
    slice_width = width // n_cols
    slice_height = height // n_rows
    
    slices = np.empty((n_rows, n_cols), dtype=object)
    contains = np.zeros((n_rows, n_cols, 6), dtype=bool)
    tree_vector = np.empty((n_rows, n_cols), dtype=object)
    tree_normals = np.empty((n_rows, n_cols), dtype=object)  # New array for normals
    building_normals = np.empty((n_rows, n_cols), dtype=object)
    building_vector = np.empty((n_rows, n_cols), dtype=object)
    direction_vectors = np.empty((n_rows, n_cols, 8), dtype=object)  # New array for 8D vectors
    
    for i in range(n_rows):
        for j in range(n_cols):
            left = j * slice_width
            upper = i * slice_height
            right = left + slice_width
            lower = upper + slice_height
            
            slice = image.crop((left, upper, right, lower))
            slices[i,j] = slice
            
            # Check for each type using colors from dictionary
            field = check_color_present(slice, COLORS['field'])
            trees = check_color_present(slice, COLORS['trees'])
            street = check_color_present(slice, COLORS['street'])
            other = check_color_present(slice, COLORS['other'])
            building = check_color_present(slice, COLORS['building'])
            water = check_color_present(slice, COLORS['water'])
            
            contains[i,j] = np.array([field, trees, street, other, building, water])
            
            if trees:
                tree_vector[i,j], tree_normals[i,j] = get_tree_edges(slice)
            else:
                tree_vector[i,j], tree_normals[i,j] = np.array([]), np.array([])
                
            if building:
                building_vector[i,j], building_normals[i,j] = get_building_edges(slice)
            else:
                building_vector[i,j], building_normals[i,j] = np.array([]), np.array([])
            
            # Generate 8D vector for this slice
            direction_vectors[i,j] = generate_direction_vector(contains[i,j],
                                                            tree_normals[i,j],
                                                            building_vector[i,j])
            
    
    return slices, contains, tree_vector, tree_normals, building_vector, building_normals, direction_vectors

def initialize(img, n, m):
    # Slice into nxm grid
    slices, contains, tree_vector, tree_normals, building_vector, building_normals, direction_vectors = slice_image(img, n, m)

    return slices, contains, tree_vector, tree_normals, building_vector, building_normals, direction_vectors

def visualize_slices(m, n,slices, contains, tree_vector, tree_normals, building_vector, building_normals, direction_vectors):
    # Display slices in a grid with spacing
    fig, axes = plt.subplots(m, n, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    # Add a text box for hover info
    hover_text = fig.text(0.5, 0.95, '', 
                        ha='center', 
                        va='center',
                        bbox=dict(facecolor='white', 
                                edgecolor='black', 
                                alpha=0.8))


    def hover(event):
        if event.inaxes in axes.flat:
            # Get the subplot index
            idx = np.where(axes.flat == event.inaxes)[0][0]
            row = idx // m
            col = idx % n
            # Update the hover text
            hover_text.set_text(f'Slice [{row},{col}]')
            fig.canvas.draw_idle()
        else:
            hover_text.set_text('')
            fig.canvas.draw_idle()

    # Connect the hover event
    fig.canvas.mpl_connect('motion_notify_event', hover)

    for i in range(m):
        for j in range(n):
            axes[i,j].imshow(slices[i,j])
            if len(tree_vector[i,j]) > 0:
                edges = tree_vector[i,j]
                normals = tree_normals[i,j]
                # Plot all edge points
                axes[i,j].plot(edges[:,0], edges[:,1], 'r.', markersize=1)
                
                # Sample every Nth point for normals (e.g., every 10th point)
                N = 10
                sampled_edges = edges[::N]
                sampled_normals = normals[::N]
                
                # Plot normals with larger arrows
                axes[i,j].quiver(sampled_edges[:,0], sampled_edges[:,1], 
                            sampled_normals[:,0], sampled_normals[:,1],
                            color='blue', scale=20,  # Reduced scale = larger arrows
                            width=0.005,  # Thicker arrows
                            headwidth=6,  # Wider arrowheads
                            headlength=9)  # Longer arrowheads
                
            if len(building_vector[i,j]) > 0:
                edges = building_vector[i,j]
                normals = building_normals[i,j]
                # Plot all edge points
                axes[i,j].plot(edges[:,0], edges[:,1], 'r.', markersize=1)
                
                # Sample every Nth point for normals visualization
                N = 10  # Adjust as needed
                sampled_edges = edges[::N]
                sampled_normals = normals[::N]
                # Plot normals with larger arrows
                axes[i,j].quiver(sampled_edges[:,0], sampled_edges[:,1], 
                            sampled_normals[:,0], sampled_normals[:,1],
                            color='red', scale=20,  # Reduced scale = larger arrows
                            width=0.005,  # Thicker arrows
                            headwidth=6,  # Wider arrowheads
                            headlength=9)  # Longer arrowheads
            axes[i,j].axis('off')
            print(f"Slice [{i},{j}] contains: {contains[i,j]}, direction vector: {direction_vectors[i,j]}")

    plt.tight_layout()
    plt.show()


def visualze_sector_probs(slices, contains, tree_vector, tree_normals, building_vector, direction_vectors, index):
    """
    Visualize slices with heatmap overlay based on direction_vectors[index]
    """
    # Get dimensions from slices
    m, n = slices.shape
    
    # Create figure with mxn subplots
    fig, axes = plt.subplots(m, n, figsize=(12, 12))

    
    # Get the probability values for the specified direction (index)
    probs = [dv[index] for dv in direction_vectors.reshape(m*n, 8)]

    
    # Find min and max values for consistent colormap scaling
    vmin = np.min(probs)
    vmax = np.max(probs)
    
    # Plot each slice with heatmap overlay
    for i in range(m):
        for j in range(n):
            # Convert PIL Image to numpy array and plot
            slice_array = np.array(slices[i,j])
            axes[i,j].imshow(slice_array)
            
            # Create a semi-transparent heatmap overlay using the probability value
            overlay = np.full(slice_array.shape[:2], direction_vectors[i,j][index])
            
            # Add the heatmap overlay
            im = axes[i,j].imshow(overlay, 
                                alpha=0.5,  # transparency
                                cmap='RdYlBu',  # Red (high) to Blue (low)
                                vmin=vmin,        # Flip the range
                                vmax=vmax)
            
            # Remove axes for cleaner look
            axes[i,j].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.title(f"Probability for sector {index}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    for i in range(4, 5):
        i = 2
    # Read image
        img = read_image(f"img/Kursk_{i}.png")

        n = 500
        m = 500

        slices, contains, tree_vector, tree_normals, building_vector, building_normals, direction_vectors = initialize(img, n, m)
        np.save(f"terrain/Kursk_{i}_{n}x{m}", direction_vectors)
        np.save(f"terrain/Kursk_{i}_{n}x{m}_roads", contains[:,:,2])

    # visualize_slices(m,n, slices, contains, tree_vector, tree_normals, building_vector, building_normals, direction_vectors)

    # index = 0
        visualze_sector_probs(slices, contains, tree_vector, tree_normals, building_vector, direction_vectors, 2)

