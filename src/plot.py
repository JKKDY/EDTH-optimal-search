from drone import Drone 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np



def animate_drone():
    # path = np.array([[0,0,2000], [5000, 5000, 2000], [5000,7000,2000], [10000,10000,2000]])
 
    # num_time_frames = 100
    # drone = Drone(path, num_timesteps=num_time_frames)

    # map_shape = (500, 500, 8)
    # pixel_size = 26
    # terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    # # Set up the figure and axis.
    # fig, ax = plt.subplots(figsize=(8, 8))

    # # Plot the entire path (assuming drone.path is a list or array of [x,y,z] points).
    # path_arr = np.array(drone.path)

    # # Compute and show the initial detection coverage.
    # # Set the extent using terrain dimensions and pixel_size.
    # extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    # # Create a marker for the drone (initially empty, but we'll update its position).
    # drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")

    # coverage = np.max(drone.detection_sphere(0, terrain, pixel_size), axis=2)
    # im = ax.imshow(coverage, origin='lower', extent=extent, cmap='viridis')
    
    # # Create an arrow (using quiver) to show the drone's direction.
    # arrow_scale = 0.2  # adjust as needed
    # pos = drone.positions[0]
    # direction = drone.directions[0]
    # quiver = ax.quiver(pos[0], pos[1],
    #                    direction[0]*arrow_scale, direction[1]*arrow_scale,
    #                    angles='xy', scale_units='xy', scale=1, color='k')

    # # Set axis labels and title.
    # ax.set_xlabel('World X')
    # ax.set_ylabel('World Y')
    # ax.set_title('Drone Camera Coverage Map')
    # ax.legend()
    # # ax.plot(path_arr[:, 0], path_arr[:, 1], 'k--', label="Path")
    # ax.set_aspect('equal', adjustable='box')

    # def update(frame):
    #     # Move the drone and update camera orientation (e.g., oscillate azimuth).        
    #     # Recalculate detection coverage.
    #     coverage = drone.detection_sphere(frame, terrain, pixel_size, pixel_size)
    #     coverage = np.max(coverage, axis=2)
    #     im.set_data(coverage)
        
    #     # Update the drone marker position:
    #     # Provide the position as sequences/lists.
    #     drone_marker.set_data([drone.position[0]], [drone.position[1]])
        
    #     # # Update the quiver arrow:
    #     quiver.set_offsets([drone.position[0], drone.position[1]])
    #     quiver.set_UVC(drone.direction[0]*arrow_scale, drone.direction[1]*arrow_scale)
        
    #     return im, drone_marker, quiver

    # # Create and show the animation.
    # ani = FuncAnimation(fig, update, frames=num_time_frames, interval=500, blit=True)
    # plt.show()
    path = np.array([[0, 0, 2000],
                     [5000, 5000, 2000],
                     [5000, 7000, 2000],
                     [10000, 10000, 2000]])
    num_time_frames = 100
    drone = Drone(path, num_timesteps=num_time_frames)

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    # Initial drone marker and coverage display.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    coverage = np.max(drone.detection_sphere(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='viridis')

    arrow_scale = 0.2
    pos = drone.positions[0]
    direction = drone.directions[0]
    quiver = ax.quiver(pos[0], pos[1],
                       direction[0]*arrow_scale, direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='k')

    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_title('Drone Camera Coverage Map')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    def update(frame):
        coverage = drone.detection_sphere(frame, terrain, pixel_size)
        coverage = np.max(coverage, axis=2)
        im.set_data(coverage)
        
        pos = drone.positions[frame]
        drone_marker.set_data([pos[0]], [pos[1]])
        
        direction = drone.directions[frame]
        quiver.set_offsets([pos[0], pos[1]])
        quiver.set_UVC(direction[0] * arrow_scale, direction[1] * arrow_scale)
        
        return im, drone_marker, quiver

    ani = FuncAnimation(fig, update, frames=num_time_frames, interval=500, blit=True)
    plt.show()





def plot_drone_view_cone():
     # Define a simple 2D path.
    # path = np.array([[0,0,10], [0, 5,10], [5,7,10], [10,10,10]])
    path = np.array([[0,0,2000], [5000, 5000, 2000], [5000,7000,2000], [10000,10000,2000]])
    # path = np.array([[0,0,2000], [9000, 2000, 2000], [1000,4000,2000], [9000,6000,2000]])
    # path = np.array([[0,0,2000], [9000, 2000, 2000]])

    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)
    # terrain = np.flip(terrain)
    # terrain[:, :, 4:] *= 0.9
    # terrain[:, :, :4] *= 0.5
    # plot_drone(drone, terrain, pixel_size, dt)

    idx = 45
    detection_coverage = drone.detection_sphere(idx, terrain, pixel_size)
    # detection_coverage = 1 - (np.prod(1-detection_coverage, axis=2))
    detection_coverage = np.sum(detection_coverage, axis=2)
    # detection_coverage = np.clip(detection_coverage[:, :], 0, 1)

    plt.figure(figsize=(6, 6), dpi=200)

    # plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    arrow_scale = 3
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='bone')
    direction = drone.directions[idx] 
    pos = drone.positions[idx] - direction
    plt.arrow(pos[0], pos[1],
            direction[0]*arrow_scale, direction[1]*arrow_scale,
            head_width=300, head_length=300, fc='#d11919', ec='#d11919', width=30)
    plt.title('Drone Camera Coverage Map')
    plt.xlabel('World X [m]')
    plt.ylabel('World Y [m]')
    plt.axis('equal')
    plt.tight_layout()
    # plt.savefig("img/drone_view_cone_color_bar.png")
    plt.show()
    




# plot_drone_view_cone()


def plot_drone_view_cone_no_terrain():
     # Define a simple 2D path.
    # path = np.array([[0,0,10], [0, 5,10], [5,7,10], [10,10,10]])
    path = np.array([[0,0,2000], [5000, 5000, 2000], [5000,7000,2000], [10000,10000,2000]])
    # path = np.array([[0,0,2000], [9000, 2000, 2000], [1000,4000,2000], [9000,6000,2000]])
    # path = np.array([[0,0,2000], [9000, 2000, 2000]])

    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain =  np.ones(map_shape)
    # terrain = np.flip(terrain)
    # terrain[:, :, 4:] *= 0.9
    # terrain[:, :, :4] *= 0.5
    # plot_drone(drone, terrain, pixel_size, dt)

    idx = 45
    detection_coverage = drone.detection_sphere(idx, terrain, pixel_size)
    # detection_coverage = 1 - (np.prod(1-detection_coverage, axis=2))
    detection_coverage = np.sum(detection_coverage, axis=2)
    # detection_coverage = np.clip(detection_coverage[:, :], 0, 1)

    plt.figure(figsize=(6, 6), dpi=200)

    # plt.plot(path[:, 0], path[:, 1], 'k--', label="Path")
    arrow_scale = 3
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='bone')
    direction = drone.directions[idx] 
    pos = drone.positions[idx] - direction
    plt.arrow(pos[0], pos[1],
            direction[0]*arrow_scale, direction[1]*arrow_scale,
            head_width=300, head_length=300, fc='#d11919', ec='#d11919', width=30)
    plt.title('Drone Camera Coverage Map')
    plt.xlabel('World X [m]')
    plt.ylabel('World Y [m]')
    plt.axis('equal')
    plt.tight_layout()
    # plt.savefig("img/drone_view_cone_no_terrain.png")
    plt.show()


def plot_terrain():
    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)
    terrain = np.max(terrain, axis = 2)
    terrain = terrain.astype(int)

    print(terrain.shape)

    plt.imshow(terrain, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='viridis')
    
    plt.xlabel('World X [m]')
    plt.ylabel('World Y [m]')
    plt.axis('equal')
    plt.tight_layout()
    # plt.savefig("img/map.png")
    plt.show()



if __name__ == "__main__":
    plot_drone_view_cone_no_terrain()
    plot_terrain()
    plot_drone_view_cone()
    animate_drone()
