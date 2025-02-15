from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from scipy import ndimage

file = "/Users/riarosenauer/Library/Mobile Documents/com~apple~CloudDocs/Ria/Bilder/Bildschirmfotos/Bildschirmfoto 2025-02-15 um 13.56.05.png"
file = "/Users/riarosenauer/Library/Mobile Documents/com~apple~CloudDocs/Ria/Bilder/Bildschirmfotos/Bildschirmfoto 2025-02-15 um 13.57.08.png"
file = "/Users/riarosenauer/Library/Mobile Documents/com~apple~CloudDocs/Ria/Bilder/Bildschirmfotos/Bildschirmfoto 2025-02-15 um 14.22.17.png"
file = "/Users/riarosenauer/Library/Mobile Documents/com~apple~CloudDocs/Ria/Bilder/Bildschirmfotos/Bildschirmfoto 2025-02-15 um 14.28.33.png"
# Color dictionary
COLORS = {
    'field': ['#edf0d4', '#cdebb0', '#87e0be', "#ecedd5", "#e6e8d0", "#f4f5dd", "#e3e6ce", "#cee5b1", "#f1eee8"],
    'trees': ['#add19d', '#c8d7aa', "#c8d3ad", "#b4d0a2", "#a0bb8e", "#a4c896", "#a0c491"],
    'street': ['#ffffff', '#f6fabf', '#a6822b', "#717171", "#dddde8", "#f6fabf", "#ac8332", "#f9fac7"],
    'other': ['#aedfa2', '#87e0be', '#f2dad9', "#ead6b8"],
    'building': ['#e0dfdf', '#f5dcba', "#d9d0c9", "#c3b5ab"],
    'water': ['#abd3de', '#cdebb0']
}

tolerance = 5

def read_image(file):
    return Image.open(file)

def check_color_present(slice_img, colors, tolerance=tolerance):
    return any(check_single_color(slice_img, color, tolerance) for color in colors)

def check_single_color(slice_img, target_color, tolerance=5):
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
                         sigma=4,        # Increase for more smoothing
                         low_threshold=0.3,
                         high_threshold=0.6)
    edge_coords = np.where(edges)
    if len(edge_coords[0]) > 0:
        return np.column_stack((edge_coords[1], edge_coords[0]))
    return np.array([])

def get_building_edges(slice_img, tolerance=tolerance + 5):
    np_slice = np.array(slice_img)
    
    # Create masks
    building_mask = np.zeros(np_slice.shape[:2], dtype=bool)
    street_mask = np.zeros(np_slice.shape[:2], dtype=bool)
    
    # Get building pixels (just the specific color we want)
    building_color = '#d7d0c9'
    r = int(building_color[1:3], 16)
    g = int(building_color[3:5], 16)
    b = int(building_color[5:7], 16)
    
    building_mask = (
        (np.abs(np_slice[:,:,0] - r) < tolerance) &
        (np.abs(np_slice[:,:,1] - g) < tolerance) &
        (np.abs(np_slice[:,:,2] - b) < tolerance)
    )
    
    # Get street pixels (just the specific color we want)
    street_color = '#dddde6'
    r = int(street_color[1:3], 16)
    g = int(street_color[3:5], 16)
    b = int(street_color[5:7], 16)
    
    street_mask = (
        (np.abs(np_slice[:,:,0] - r) < tolerance) &
        (np.abs(np_slice[:,:,1] - g) < tolerance) &
        (np.abs(np_slice[:,:,2] - b) < tolerance)
    )
    
    
    # Simple edge detection
    edges = feature.canny(building_mask.astype(float), sigma=3, low_threshold=0.9, high_threshold=1)
    
    edge_coords = np.where(edges)
    if len(edge_coords[0]) > 0:
        return np.column_stack((edge_coords[1], edge_coords[0]))
    return np.array([])

def slice_image(image, n_rows, n_cols):
    width, height = image.size
    slice_width = width // n_cols
    slice_height = height // n_rows
    
    slices = np.empty((n_rows, n_cols), dtype=object)
    contains = np.empty((n_rows, n_cols), dtype=object)
    tree_vector = np.empty((n_rows, n_cols), dtype=object)
    building_vector = np.empty((n_rows, n_cols), dtype=object)
    
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
                tree_vector[i,j] = get_tree_edges(slice)
            else:
                tree_vector[i,j] = np.array([])
                
            if building:
                building_vector[i,j] = get_building_edges(slice)
            else:
                building_vector[i,j] = np.array([])
    
    return slices, contains, tree_vector, building_vector

# Read image
img = read_image(file)

#plt.imshow(img)
#plt.show()

n = 30
m = 30

# Slice into 3x3 grid
slices, contains, tree_vector, building_vector = slice_image(img, n, m)

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
            axes[i,j].plot(edges[:,0], edges[:,1], 'r.', markersize=1, alpha=0.5)
        if len(building_vector[i,j]) > 0:
            edges = building_vector[i,j]
            axes[i,j].plot(edges[:,0], edges[:,1], 'b.', markersize=1, alpha=0.5)
        axes[i,j].axis('off')
        print(f"Slice [{i},{j}] contains: {contains[i,j]}")

plt.tight_layout()
plt.show()


