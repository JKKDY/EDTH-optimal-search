from drone import Drone 
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import numpy as np



def animate_drone():
    path = np.array([[0, 0, 2000],
                     [5000, 5000, 2000],
                     [5000, 7000, 2000],
                     [10000, 10000, 2000]])
    num_time_frames = 200
    drone = Drone(path, num_timesteps=num_time_frames)

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    # Initial drone marker and coverage display.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    coverage = np.max(drone.coverage_of_drone(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='bone')

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
        coverage = drone.coverage_of_drone(frame, terrain, pixel_size)
        coverage = np.max(coverage, axis=2)
        im.set_data(coverage)
        
        pos = drone.positions[frame]
        drone_marker.set_data([pos[0]], [pos[1]])
        
        direction = drone.directions[frame]
        quiver.set_offsets([pos[0], pos[1]])
        quiver.set_UVC(direction[0] * arrow_scale, direction[1] * arrow_scale)
        
        return im, drone_marker, quiver

    ani = FuncAnimation(fig, update, frames=num_time_frames, interval=5, blit=True)
    plt.show()




def animate_drone_cumulative():
    path = np.array([[0, 0, 2000],
                     [5000, 5000, 2000],
                     [5000, 7000, 2000],
                     [10000, 10000, 2000]])
    num_time_frames = 200
    drone = Drone(path, num_timesteps=num_time_frames)

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    # Initial drone marker and coverage display.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    coverage = np.max(drone.coverage_of_drone(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='bone')

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

    cumulative_coverage = np.zeros((terrain.shape[0], terrain.shape[1]))

    def update(frame):
        nonlocal cumulative_coverage  # Allow modification of outer scope variable

        cov = drone.coverage_of_drone(frame, terrain, pixel_size)
        cov = np.sum(cov, axis=2)  # Aggregate along the third dimension

        # Update cumulative coverage
        cumulative_coverage = np.maximum(cumulative_coverage, cov)

        im.set_data(cumulative_coverage)

        pos = drone.positions[frame]
        drone_marker.set_data([pos[0]], [pos[1]])

        direction = drone.directions[frame]
        quiver.set_offsets([pos[0], pos[1]])
        quiver.set_UVC(direction[0] * arrow_scale, direction[1] * arrow_scale)

        return im, drone_marker, quiver

    ani = FuncAnimation(fig, update, frames=num_time_frames, interval=5, blit=True)
    plt.show()


def plot_drone_view_cone():
    path = np.array([[0,0,2000], [5000, 5000, 2000], [5000,7000,2000], [10000,10000,2000]])

    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    idx = 45
    detection_coverage = drone.coverage_of_drone(idx, terrain, pixel_size)
    detection_coverage = np.sum(detection_coverage, axis=2)

    plt.figure(figsize=(6, 6), dpi=200)

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
    path = np.array([[0,0,2000], [5000, 5000, 2000], [5000,7000,2000], [10000,10000,2000]])
    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain =  np.ones(map_shape)

    idx = 60
    print(drone.positions[idx])
    detection_coverage = drone.coverage_of_drone(idx, terrain, pixel_size)
    detection_coverage = np.sum(detection_coverage, axis=2)

    plt.figure(figsize=(6, 6), dpi=200)

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


def plot_drone_view_camera_no_terrain():
    path = np.array([[0,0,2000], [7000, 5000, 2000], [7000,5000,2000], [10000,10000,2000]])

    drone = Drone(path, camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0), max_detection_distance=5000)

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain =  np.ones(map_shape)

    idx = 50
    print(drone.positions[idx]/pixel_size)
    detection_coverage = drone.coverage_of_drone(idx, terrain, pixel_size)
    detection_coverage = np.sum(detection_coverage, axis=2)
    print(np.unravel_index(np.argmax(detection_coverage), detection_coverage.shape))
    plt.figure(figsize=(6, 6), dpi=200)

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
            cmap='bone')
    
    plt.xlabel('World X [m]')
    plt.ylabel('World Y [m]')
    plt.axis('equal')
    plt.tight_layout()
    # plt.savefig("img/map.png")
    plt.show()



if __name__ == "__main__":
    plot_terrain()
    plot_drone_view_cone()
    animate_drone()
    animate_drone_cumulative()
