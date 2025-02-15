import tkinter as tk
from tkinter import ttk
import folium
import webbrowser
import os
from geopy.geocoders import Nominatim
import math
import osmnx as ox
from shapely.geometry import box
import tempfile

class MapViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Map Viewer")
        
        # Initialize variables
        self.lat = tk.DoubleVar(value=48.137154)  # Munich coordinates
        self.lon = tk.DoubleVar(value=11.576124)
        self.location_name = tk.StringVar()
        self.zoom_level = 15
        
        self.create_gui()
        
    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Coordinates input
        ttk.Label(main_frame, text="Coordinates Input").grid(row=0, column=0, columnspan=2)
        
        ttk.Label(main_frame, text="Latitude:").grid(row=1, column=0)
        ttk.Entry(main_frame, textvariable=self.lat).grid(row=1, column=1)
        
        ttk.Label(main_frame, text="Longitude:").grid(row=2, column=0)
        ttk.Entry(main_frame, textvariable=self.lon).grid(row=2, column=1)
        
        # Location name input
        ttk.Label(main_frame, text="Location Name:").grid(row=3, column=0)
        location_entry = ttk.Entry(main_frame, textvariable=self.location_name)
        location_entry.grid(row=3, column=1)
        
        # Buttons
        ttk.Button(main_frame, text="Show Satellite View", 
                  command=lambda: self.show_map("Satellite")).grid(row=4, column=0)
        ttk.Button(main_frame, text="Show OSM View", 
                  command=lambda: self.show_map("OSM")).grid(row=4, column=1)
        ttk.Button(main_frame, text="Search Location", 
                  command=self.search_location).grid(row=5, column=0, columnspan=2)
        
    def get_features_for_area(self, center_lat, center_lon):
        """Get OSM features for the area"""
        try:
            # Calculate a smaller bounding box (100m x 100m)
            distance = 50  # meters from center point
            north = center_lat + (distance / 111111)
            south = center_lat - (distance / 111111)
            east = center_lon + (distance / (111111 * math.cos(math.radians(center_lat))))
            west = center_lon - (distance / (111111 * math.cos(math.radians(center_lat))))
            
            bbox = (north, south, east, west)
            
            tags = {
                'natural': ['water', 'tree', 'wood'],
                'waterway': True,
                'landuse': ['forest'],
                'highway': True,
                'building': True
            }
            
            features = ox.features.features_from_bbox(bbox=bbox, tags=tags)
            
            return {
                'water': features[features['waterway'].notna() | 
                                (features['natural'] == 'water')],
                'trees': features[features['natural'].isin(['tree', 'wood']) | 
                                (features['landuse'] == 'forest')],
                'street': features[features['highway'].notna()],
                'building': features[features['building'].notna()]
            }
            
        except Exception as e:
            print(f"Error fetching OSM data: {e}")
            return None
    
    def create_map(self, map_type="OSM"):
        """Create map with features"""
        # Create base map
        m = folium.Map(
            location=[self.lat.get(), self.lon.get()],
            zoom_start=self.zoom_level
        )
        
        if map_type == "Satellite":
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite'
            ).add_to(m)
        
        # Add features grid
        features_dict = self.get_features_for_area(self.lat.get(), self.lon.get())
        if features_dict:
            self.add_features_to_map(m, features_dict)
        
        # Add marker for current location
        folium.Marker([self.lat.get(), self.lon.get()]).add_to(m)
        
        return m
    
    def add_features_to_map(self, m, features_dict):
        """Add feature grid to map"""
        # Similar to the create_grid function from before
        # ... (grid creation code) ...
        pass
    
    def show_map(self, map_type):
        """Display the map in browser"""
        m = self.create_map(map_type)
        
        # Save map to temporary file and open in browser
        _, temp_file = tempfile.mkstemp(suffix='.html')
        m.save(temp_file)
        webbrowser.open('file://' + temp_file)
    
    def search_location(self):
        """Search location by name"""
        try:
            geolocator = Nominatim(user_agent="map_viewer")
            location = geolocator.geocode(self.location_name.get())
            
            if location:
                self.lat.set(location.latitude)
                self.lon.set(location.longitude)
            else:
                print("Location not found")
                
        except Exception as e:
            print(f"Error finding location: {e}")

def main():
    root = tk.Tk()
    app = MapViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()