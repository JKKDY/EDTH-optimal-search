import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from drone import Drone 
from pathing import circular_path


def prepare_environment(path=None, n_points = 300):
    if path is None: 
        path = np.array([[0, 0, 2000],
                     [5000, 5000, 2000],
                     [5000, 7000, 2000],
                     [10000, 10000, 2000]])
    num_time_frames = n_points
    drone = Drone(path, num_timesteps=num_time_frames)

    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)

    return drone, terrain, pixel_size



def prepare_ax(ax):
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.set_title('Drone Camera Coverage Map')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')



def prepare_plt():
    plt.title('Drone Camera Coverage Map')
    plt.xlabel('World X [m]')
    plt.ylabel('World Y [m]')
    plt.axis('equal')
    plt.tight_layout()

def animate_drone_coverage():
    drone, terrain, pixel_size = prepare_environment()

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    # Initial drone marker and coverage display.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    coverage = np.max(drone.coverage_of_drone(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='bone')

    arrow_scale = 10
    pos = drone.positions[0]
    direction = drone.directions[0]
    quiver = ax.quiver(pos[0], pos[1],
                       direction[0]*arrow_scale, direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='#d11919')

    prepare_ax(ax)
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

    ani = FuncAnimation(fig, update, frames=drone.num_timesteps, interval=5, blit=True)
    ani.save('img/animate_drone_coverage.gif', writer='pillow', fps=40)
    plt.show()




def animate_drone_coverage_cumulative():
    circular_path
    drone, terrain, pixel_size = prepare_environment()

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    # Initial drone marker and coverage display.
    drone_marker, = ax.plot([], [], 'ro', markersize=8, label="Drone")
    coverage = np.max(drone.coverage_of_drone(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='bone')

    arrow_scale = 10
    pos = drone.positions[0]
    direction = drone.directions[0]
    quiver = ax.quiver(pos[0], pos[1],
                       direction[0]*arrow_scale, direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='#d11919')

    cumulative_coverage = np.zeros((terrain.shape[0], terrain.shape[1]))
    prepare_ax(ax)

    def update(frame):
        nonlocal cumulative_coverage  # Allow modification of outer scope variable

        cov = drone.coverage_of_drone(frame, terrain, pixel_size)
        cov = np.sum(cov, axis=2) 
        cumulative_coverage = np.maximum(cumulative_coverage, cov)

        im.set_data(cumulative_coverage)

        pos = drone.positions[frame]
        drone_marker.set_data([pos[0]], [pos[1]])

        direction = drone.directions[frame]
        quiver.set_offsets([pos[0], pos[1]])
        quiver.set_UVC(direction[0] * arrow_scale, direction[1] * arrow_scale)

        return im, drone_marker, quiver

    ani = FuncAnimation(fig, update, frames=drone.num_timesteps, interval=5, blit=True)
    ani.save('img/animate_drone_coverage_cumulative.gif', writer='pillow', fps=40)
    plt.show()



def animate_drone_camera_coverage():
    path = circular_path((6000, 6000), 5000, 300)
    drone, terrain, pixel_size = prepare_environment(path, 1000)
    drone.adjust_camera(camera_fov=np.deg2rad(60), camera_elevation=np.deg2rad(45))

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    drone_marker, = ax.plot([], [], 'ro', markersize=2, label="Drone")
    coverage = np.max(drone.coverage_of_drone(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='bone')

    arrow_scale = 10
    pos = drone.positions[0]
    direction = drone.directions[0]
    quiver = ax.quiver(pos[0], pos[1],
                       direction[0]*arrow_scale, direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='#d11919')

    prepare_ax(ax)
    def update(frame):
        nonlocal terrain
        drone.adjust_camera(camera_azimuth=0.6*np.sin(frame/10) + np.deg2rad(40), camera_fov = 0.1 * np.sin(frame/30) + np.deg2rad(30),
                            camera_elevation=0.2*np.sin(frame/30) + np.deg2rad(30))
        cov = drone.coverage_of_camera(frame, terrain, pixel_size)
        cov = np.sum(cov, axis=2) 
        im.set_data(cov + 0.1*np.max(terrain, axis=2).astype(np.float64))

        pos = drone.positions[frame]
        drone_marker.set_data([pos[0]], [pos[1]])

        direction = drone.directions[frame]
        quiver.set_offsets([pos[0], pos[1]])
        quiver.set_UVC(direction[0] * arrow_scale, direction[1] * arrow_scale)

        return im, drone_marker, quiver

    ani = FuncAnimation(fig, update, frames=drone.num_timesteps, interval=10, blit=True)
    ani.save('img/animate_drone_camera_coverage.gif', writer='pillow', fps=40)
    plt.show()


def animate_drone_camera_coverage_cummulative():
    path = circular_path((6000, 6000), 5000, 200)
    drone, terrain, pixel_size = prepare_environment(path, 1000)
    drone.adjust_camera(camera_fov=np.deg2rad(60), camera_elevation=np.deg2rad(45))

    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size]

    drone_marker, = ax.plot([], [], 'ro', markersize=2, label="Drone")
    coverage = np.max(drone.coverage_of_drone(0, terrain, pixel_size), axis=2)
    im = ax.imshow(coverage, origin='lower', extent=extent, cmap='bone')

    arrow_scale = 10
    pos = drone.positions[0]
    direction = drone.directions[0]
    quiver = ax.quiver(pos[0], pos[1],
                       direction[0]*arrow_scale, direction[1]*arrow_scale,
                       angles='xy', scale_units='xy', scale=1, color='#d11919')

    cumulative_coverage = np.zeros((terrain.shape[0], terrain.shape[1]))
    prepare_ax(ax)

    def update(frame):
        nonlocal cumulative_coverage
        drone.adjust_camera(camera_azimuth=0.6*np.sin(frame/10) + np.deg2rad(50), camera_fov = 0.1 * np.sin(frame/30) + np.deg2rad(30),
                            camera_elevation=0.2*np.sin(frame/30) + np.deg2rad(30))
        cov = drone.coverage_of_camera(frame, terrain, pixel_size)
        cov = np.sum(cov, axis=2) 
        cumulative_coverage = np.maximum(cumulative_coverage, cov)

        im.set_data(cumulative_coverage+ 0.1*np.max(terrain, axis=2).astype(np.float64))

        pos = drone.positions[frame]
        drone_marker.set_data([pos[0]], [pos[1]])

        direction = drone.directions[frame]
        quiver.set_offsets([pos[0], pos[1]])
        quiver.set_UVC(direction[0] * arrow_scale, direction[1] * arrow_scale)

        return im, drone_marker, quiver

    ani = FuncAnimation(fig, update, frames=drone.num_timesteps, interval=10, blit=True)
    ani.save('img/animate_drone_camera_coverage_cummulative.gif', writer='pillow', fps=40)
    plt.show()



def plot_drone_view_cone():
    drone, terrain, pixel_size = prepare_environment()

    idx = 45
    detection_coverage = drone.coverage_of_drone(idx, terrain, pixel_size)
    detection_coverage = np.sum(detection_coverage, axis=2)

    plt.figure(figsize=(6, 6), dpi=200)

    arrow_scale = 3
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size],
            cmap='bone')
    direction = drone.directions[idx] 
    pos = drone.positions[idx] - direction
    plt.arrow(pos[0], pos[1],
            direction[0]*arrow_scale, direction[1]*arrow_scale,
            head_width=300, head_length=300, fc='#d11919', ec='#d11919', width=30)
    
    prepare_plt()
    plt.savefig("img/plot_drone_view_cone.png")
    plt.show()


def plot_drone_view_cone_no_terrain():
    drone, terrain, pixel_size = prepare_environment()
    drone.adjust_camera(camera_elevation=np.deg2rad(45), camera_fov=np.deg2rad(60), camera_azimuth=np.deg2rad(0))
    terrain =  np.ones(terrain.shape)

    idx = 60
    detection_coverage = drone.coverage_of_drone(idx, terrain, pixel_size)
    detection_coverage = np.sum(detection_coverage, axis=2)

    plt.figure(figsize=(6, 6), dpi=200)

    arrow_scale = 3
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size],
            cmap='bone')
    direction = drone.directions[idx] 
    pos = drone.positions[idx] - direction
    plt.arrow(pos[0], pos[1],
            direction[0]*arrow_scale, direction[1]*arrow_scale,
            head_width=300, head_length=300, fc='#d11919', ec='#d11919', width=30)
    
    prepare_plt()
    plt.savefig("img/plot_drone_view_cone_no_terrain.png")
    plt.show()


def plot_drone_view_camera_no_terrain():
    drone, terrain, pixel_size = prepare_environment()

    idx = 50
    detection_coverage = drone.coverage_of_drone(idx, terrain, pixel_size)
    detection_coverage = np.sum(detection_coverage, axis=2)
    plt.figure(figsize=(6, 6), dpi=200)

    arrow_scale = 3
    plt.imshow(detection_coverage, origin="lower",
            extent=[0, terrain.shape[1]*pixel_size, 0, terrain.shape[0]*pixel_size],
            cmap='bone')
    direction = drone.directions[idx] 
    pos = drone.positions[idx] - direction
    plt.arrow(pos[0], pos[1],
            direction[0]*arrow_scale, direction[1]*arrow_scale,
            head_width=300, head_length=300, fc='#d11919', ec='#d11919', width=30)
    
    prepare_plt()
    plt.savefig("img/plot_drone_view_camera_no_terrain.png")
    plt.show()


def plot_terrain():
    map_shape = (500, 500, 8)
    pixel_size = 26
    terrain = np.load("terrain/Kursk_4_500x500.npy", allow_pickle=True).reshape(map_shape)
    terrain = np.max(terrain, axis = 2)
    terrain = terrain.astype(int)

    plt.imshow(terrain, origin="lower",
            extent=[0, map_shape[1]*pixel_size, 0, map_shape[0]*pixel_size],
            cmap='bone')
    
    prepare_plt()
    plt.savefig("img/terrain.png")
    plt.show()



if __name__ == "__main__":
    animate_drone_coverage()
    animate_drone_coverage_cumulative()
    animate_drone_camera_coverage()
    animate_drone_camera_coverage_cummulative()
