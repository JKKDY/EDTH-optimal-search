import streamlit as st
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import re
import math
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
from PIL import Image


##
#install:
#pip install streamlit
#pip install Pillow 
#pip install folium
#pip install streamlit-folium
#pip install geopy
#pip install osmnx
#pip install geopandas
#pip install shapely
#pip install opencv-python


# Set page config
st.set_page_config(
    page_title="Satellite View Finder",
    page_icon="🛰️",
    layout="wide"
)

# Title and description
st.title("🛰️ Satellite View Finder")
st.write("Enter coordinates or a location name to view satellite imagery")

# Create two columns for input methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Enter Coordinates")
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.0)
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=0.0)

with col2:
    st.subheader("Option 2: Enter Location Name")
    location_name = st.text_input("Location (e.g., 'Paris, France')", value="Munich")
    
    if location_name:
        try:
            # Initialize geolocator
            geolocator = Nominatim(user_agent="satellite_viewer")
            location = geolocator.geocode(location_name)
            
            if location:
                lat = location.latitude
                lon = location.longitude
                st.success(f"Found coordinates: {lat:.4f}, {lon:.4f}")
            else:
                st.error("Location not found. Please try a different name.")
        except Exception as e:
            st.error(f"Error finding location: {str(e)}")

# Define the speed and time
speed_kmh = 80
hours = 1.5
distance_km = speed_kmh * hours  # 120km one-way distance

def calculate_zoom_for_square_distance(one_way_distance_km, latitude):
    """
    Calculate zoom level for a square map where the distance from center to edge is one_way_distance_km
    Using a more accurate calculation based on real-world testing
    """
    # The map width needs to be twice the one-way distance
    full_width_km = 2 * one_way_distance_km
    
    # Adjusted formula based on real-world testing
    # At zoom level 11, the width is approximately 250km at mid-latitudes
    base_width_at_zoom_11 = 250
    zoom = 11 + math.log2(base_width_at_zoom_11 / full_width_km)
    
    return int(zoom + 0.5)  # Round to nearest integer

# Calculate the zoom level based on the distance and current latitude
calculated_zoom = calculate_zoom_for_square_distance(distance_km, lat)

# Add zoom level slider with calculated default and minimum
zoom_level = st.slider("Zoom level", 
                      min_value=calculated_zoom,
                      max_value=18, 
                      value=calculated_zoom,
                      help=f"Default zoom shows {distance_km}km in each direction from center (1.5 hours at {speed_kmh}km/h)")

# Display the current coverage info
st.write(f"At current zoom level, the view shows approximately {distance_km}km in each direction from the center point")

# Create and display map
if st.button("Show Map") or location_name:
    try:
        # Create folium map with satellite tiles and min zoom restriction
        m = folium.Map(
            location=[lat, lon],
            zoom_start=zoom_level,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            min_zoom=calculated_zoom,
        )
        
        # Add marker
        folium.Marker(
            [lat, lon],
            popup=f'Lat: {lat:.4f}, Long: {lon:.4f}'
        ).add_to(m)
        
        # Display map using streamlit-folium with full width and square dimensions
        st.components.v1.html(m._repr_html_(), height=1700, width=1700)
        
        # Display coordinates
        st.info(f"📍 Current coordinates: Latitude {lat:.4f}, Longitude {lon:.4f}")
        
    except Exception as e:
        st.error(f"Error displaying map: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and OpenStreetMap")

class SectorAnalyzer:
    def __init__(self):
        self.classes = {
            'field': {'color': (120, 120, 0)},    # brown
            'trees': {'color': (0, 255, 0)},      # green
            'street': {'color': (128, 128, 128)}, # gray
            'water': {'color': (0, 0, 255)},      # blue
            'building': {'color': (200, 200, 200)}, # light gray
            'other': {'color': (100, 100, 100)}   # dark gray
        }
    
    def create_grid(self, image, num_rows=30, num_cols=40):
        """Create a grid of sectors"""
        height, width = image.shape[:2]
        row_height = height // num_rows
        col_width = width // num_cols
        
        sectors = []
        for i in range(num_rows):
            for j in range(num_cols):
                sector = {
                    'bounds': (
                        j * col_width,
                        i * row_height,
                        (j + 1) * col_width,
                        (i + 1) * row_height
                    ),
                    'id': f'sector_{i}_{j}'
                }
                sectors.append(sector)
        return sectors

    def analyze_sector(self, image, sector):
        """Analyze contents of a single sector"""
        x1, y1, x2, y2 = sector['bounds']
        sector_img = image[y1:y2, x1:x2]
        
        results = {
            'field': 0,
            'trees': 0,
            'street': 0,
            'water': 0,
            'building': 0,
            'other': 0
        }
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(sector_img, cv2.COLOR_RGB2HSV)
        
        # Detect green (trees)
        green_mask = cv2.inRange(hsv, (35, 25, 25), (85, 255, 255))
        green_ratio = np.count_nonzero(green_mask) / (green_mask.shape[0] * green_mask.shape[1])
        
        # Detect blue (water)
        blue_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        blue_ratio = np.count_nonzero(blue_mask) / (blue_mask.shape[0] * blue_mask.shape[1])
        
        # Detect gray (streets)
        street_mask = cv2.inRange(hsv, (0, 0, 50), (180, 30, 150))
        street_ratio = np.count_nonzero(street_mask) / (street_mask.shape[0] * street_mask.shape[1])
        
        # Detect light gray (buildings)
        building_mask = cv2.inRange(hsv, (0, 0, 150), (180, 30, 250))
        building_ratio = np.count_nonzero(building_mask) / (building_mask.shape[0] * building_mask.shape[1])
        
        # Calculate field ratio (areas with brown/beige colors)
        field_mask = cv2.inRange(hsv, (20, 30, 100), (35, 255, 255))
        field_ratio = np.count_nonzero(field_mask) / (field_mask.shape[0] * field_mask.shape[1])
        
        # Assign detected ratios
        results['trees'] = green_ratio
        results['water'] = blue_ratio
        results['street'] = street_ratio
        results['building'] = building_ratio
        results['field'] = field_ratio
        
        # Calculate 'other' as remaining unclassified area
        total_classified = sum(results.values())
        results['other'] = max(0, 1 - total_classified)
        
        # Normalize results if total is > 1
        total = sum(results.values())
        if total > 1:
            for key in results:
                results[key] /= total
        
        return results

def main():
    st.title("Sector Analysis Tool")
    
    # Default file path
    default_file = "/Users/riarosenauer/Library/Mobile Documents/com~apple~CloudDocs/Ria/Bilder/Bildschirmfotos/Bildschirmfoto 2025-02-14 um 20.54.40.png"
    
    try:
        image = Image.open(default_file)
        image_np = np.array(image)
        display_image = image_np.copy()
        
        analyzer = SectorAnalyzer()
        sectors = analyzer.create_grid(image_np)
        
        # Analyze each sector
        for sector in sectors:
            results = analyzer.analyze_sector(image_np, sector)
            x1, y1, x2, y2 = sector['bounds']
            
            # Draw sector boundaries
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # Create simple text with just first letters
            text = ""
            for key, value in results.items():
                if value > 0.15:  # Threshold for showing the category
                    text += f"{key[0].upper()}"  # Just the first letter
            
            # Calculate center of sector
            sector_center_x = (x1 + x2) // 2
            sector_center_y = (y1 + y2) // 2
            
            # Draw white text
            font_scale = 1.0  # Adjusted for multiple letters
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            text_x = sector_center_x - (text_width // 2)
            text_y = sector_center_y + (text_height // 2)
            
            cv2.putText(
                display_image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        st.image(display_image, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()