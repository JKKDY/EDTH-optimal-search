from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

file = "/Users/riarosenauer/Library/Mobile Documents/com~apple~CloudDocs/Ria/Bilder/Bildschirmfotos/Bildschirmfoto 2025-02-15 um 12.26.32.png"


def read_image(file):
    return Image.open(file)

def check_color_present(slice_img, target_color, tolerance=30):
    # Convert slice to numpy array
    np_slice = np.array(slice_img)
    # Convert hex to RGB
    r = int(target_color[1:3], 16)
    g = int(target_color[3:5], 16)
    b = int(target_color[5:7], 16)
    
    # Check if any pixel is within tolerance of target color
    color_present = np.any(
        (np.abs(np_slice[:,:,0] - r) < tolerance) &
        (np.abs(np_slice[:,:,1] - g) < tolerance) &
        (np.abs(np_slice[:,:,2] - b) < tolerance)
    )
    return color_present

def slice_image(image, n_rows, n_cols):
    width, height = image.size
    slice_width = width // n_cols
    slice_height = height // n_rows
    
    slices = np.empty((n_rows, n_cols), dtype=object)
    contains = np.empty((n_rows, n_cols), dtype=object)
    building_vector = np.empty((n_rows, n_cols), dtype=object)
    tree_vector = np.empty((n_rows, n_cols), dtype=object)
    other_vector = np.empty((n_rows, n_cols), dtype=object)
    
    for i in range(n_rows):
        for j in range(n_cols):
            left = j * slice_width
            upper = i * slice_height
            right = left + slice_width
            lower = upper + slice_height
            
            slice = image.crop((left, upper, right, lower))
            slices[i,j] = slice
            
            # Check for each color and create boolean array
            field = check_color_present(slice, '#edf0d4') or check_color_present(slice, '#cdebb0') or check_color_present(slice, '#87e0be')
            trees = check_color_present(slice, '#add19d') or check_color_present(slice, '#c8d7aa')
            street = check_color_present(slice, '#ffffff') or check_color_present(slice, '#f6fabf') or check_color_present(slice, '#a6822b')
            other = check_color_present(slice, '#aedfa2') or check_color_present(slice, '#87e0be') or check_color_present(slice, '#f2dad9')
            building = check_color_present(slice, '#e0dfdf') or check_color_present(slice, '#f5dcba')
            water = check_color_present(slice, '#abd3de') or check_color_present(slice, '#cdebb0')
            
            contains[i,j] = np.array([field, trees, street, other, building, water])
    
    return slices, contains

# Read image
img = read_image(file)

#plt.imshow(img)
#plt.show()

n = 12
m = 12

# Slice into 3x3 grid
slices, contains = slice_image(img, n, m)

# Display slices in a grid with spacing
fig, axes = plt.subplots(n, m, figsize=(8, 8))
plt.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(n):
    for j in range(m):
        axes[i,j].imshow(slices[i,j])
        axes[i,j].axis('off')
        print(f"Slice [{i},{j}] contains: {contains[i,j]}")

plt.tight_layout()
plt.show()

