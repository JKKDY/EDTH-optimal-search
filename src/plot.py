from drone import drone 
import matplotlib.pyplot as plt
import numpy as np



def animate_drone_(drone, terrain, pixel_size, dt):
    """
    Create an animation of the drone's flight with updated detection coverage.

    Parameters:
      drone      : an instance of Drone.
      terrain    : 3D numpy array representing the terrain (shape: [rows, cols, channels]).
      pixel_size : scalar, size of a pixel in world units.
      dt         : time step duration for each frame.
    """
    # Set up the figure and axis.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the entire path (assuming drone.path is a list or array of [x,y,z] points).
    path_arr = np.array(drone.path)

    # Compute and show the initial detection coverage.
    initial_coverage = drone.detection_coverage(terrain, pixel_size, certain_detection_distance=4, max_detection_distance=10)
    # Set the extent using terrain dimensions and pixel_size.
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]
    im = ax.imshow(initial_coverage, origin='lower', extent=extent, cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, label='Detection Probability')

    # Create a marker for the drone (initially empty, but we'll update its position).
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    
    # Create an arrow (using quiver) to show the drone's direction.
    arrow_scale = 0.2  # adjust as needed
    quiver = ax.quiver(drone.position[0], drone.position[1],
                       drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='k')

    # Set axis labels and title.
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_title('Drone Camera Coverage Map')
    ax.legend()
    ax.plot(path_arr[:, 0], path_arr[:, 1], 'k--', label="Path")
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        # Move the drone and update camera orientation (e.g., oscillate azimuth).
        drone.move(dt)
        drone.adjust_camera(azimuth=0.6*np.sin(frame*0.1))
        
        # Recalculate detection coverage.
        coverage = drone.detection_coverage(terrain, pixel_size, certain_detection_distance=4, max_detection_distance=10)
        detection_coverage = np.sum(detection_coverage, axis=2)

        im.set_data(coverage)
        
        # Update the drone marker position:
        # Provide the position as sequences/lists.
        drone_marker.set_data([drone.position[0]], [drone.position[1]])
        
        # Update the quiver arrow:
        quiver.set_offsets([drone.position[0], drone.position[1]])
        quiver.set_UVC(drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale)
        
        return im, drone_marker, quiver

    # Create and show the animation.
    ani = FuncAnimation(fig, update, frames=300, interval=dt*500, blit=True)
    plt.show()