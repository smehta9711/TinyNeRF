import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pynput import keyboard
import threading
import torch
import torch.nn as nn
from nerf_model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Initial camera parameters
radius = 3.5
angle_deg = 0  # In degrees, around the Y-axis (XZ-plane circle)
z_height = 2.2  # Initial z-height
angle_step = 5  # degrees
height_step = 0.2
radius_step = 0.15

# Bounds
z_min, z_max = 0,100
radius_min, radius_max = 0,100

# Origin (target point camera looks at)
origin = np.array([0, 0, 0])

def test_nerf(checkpoint_path,c2w, output_dir='novel_views', 
              chunk_size=10, nb_bins=192, H=400, W=400, device='cuda',
              start_idx=0, end_idx=None, hn=2, hf=6):

   
    
    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract checkpoint parameters
    hn = checkpoint.get('hn', hn)
    hf = checkpoint.get('hf', hf)
    nb_bins = checkpoint.get('nb_bins', nb_bins)
    
    # Initialize the model
   
    model = NerfModel(hidden_dim=256).to(device)
    if checkpoint_path=="./nerf_model_final.pth":
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()  # Set to evaluation mode
    
  
 

    data = []  # list of regenerated pixel values
    
    H, W = 400, 400
    chunk_size = 2  # Assuming you're processing 10 rows at a time
    data = []

    for h_start in range(0, H, chunk_size):
        h_end = min(h_start + chunk_size, H)
        current_chunk_size = h_end - h_start

        ray_origins_np = np.zeros((current_chunk_size, W, 3), dtype=np.float32)
        ray_directions_np = np.zeros((current_chunk_size, W, 3), dtype=np.float32)

        for dh in range(current_chunk_size):
            h = h_start + dh
            for w in range(W):
                o, d = get_ray_pixel(h, w, c2w)
                ray_origins_np[dh, w, :] = o
                ray_directions_np[dh, w, :] = d

        # Convert once per chunk (not inside pixel loop)
        ray_origins = torch.from_numpy(ray_origins_np.reshape(-1, 3)).float().to(device)
        ray_directions = torch.from_numpy(ray_directions_np.reshape(-1, 3)).float().to(device)

        with torch.no_grad():
            regenerated_px_values = render_rays(model, ray_origins, ray_directions,
                                                hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)

    # Reconstruct full image
    img = torch.cat(data, dim=0).data.cpu().numpy().reshape(H, W, 3)

    
    # Save the rendered image
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(f'{output_dir}/rendered_img.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # print(f"Saved image {img_index} to {output_dir}/img_{img_index}.png")

#assume now we have k
##yerlin method
def get_ray_pixel(h, w, c2w):
    """
    Computes the ray origin and direction for a single pixel (h, w).
    
    Args:
        h (int): Pixel row index.
        w (int): Pixel column index.
        K (numpy.ndarray): 3x3 intrinsic camera matrix.
        c2w (numpy.ndarray): 4x4 camera-to-world transformation matrix.
    
    Returns:K = camera_info["intrinsics"]
        rays_o (numpy.ndarray): The origin of the ray (3,)
        rays_d (numpy.ndarray): The direction of the ray (3,)
    """
    # Convert pixel coordinates to normalized camera coordinates
    focal=535.81
    i = (w - 200 ) /focal
    j = -(h -200) / focal
    
    # Ray direction in camera frame
    dir_cam = np.array([i, j, -1.0])  # Assume a pinhole camera model
    
    # Transform ray direction to world frame
    dir_world = c2w[:3, :3] @ dir_cam  # Rotate using the camera-to-world rotation matrix
    
    # Normalize the direction vector
    dir_world /= np.linalg.norm(dir_world)
    
    # The origin of the ray is the camera's position in world coordinates
    origin_world = c2w[:3, 3]
    
    return origin_world, dir_world


def compute_frame(translation):
    convec = origin - translation
    convec /= np.linalg.norm(convec)
    helper = np.array([1, 0, 0]) if abs(np.dot(convec, [0, 0, 1])) > 0.99 else np.array([0, 0, 1])
    v_in_p1 = helper - np.dot(helper, convec) * convec
    v_in_p1 /= np.linalg.norm(v_in_p1)
    xvec = np.cross(convec, v_in_p1)
    xvec /= np.linalg.norm(xvec)
    yvec = np.cross(xvec, convec)
    yvec /= np.linalg.norm(yvec)
    convec = -convec  # Forward vector
    return np.column_stack([xvec, yvec, convec])

def update_plot(ax, translation, R_local_to_world):
    ax.clear()

    # World frame
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', linestyle='dashed', length=0.5, normalize=True, label='World X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', linestyle='dashed', length=0.5, normalize=True, label='World Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', linestyle='dashed', length=0.5, normalize=True, label='World Z')

    # Local frame
    ax.quiver(*translation, *R_local_to_world[:, 0], color='r', length=1, normalize=True, label='Local xvec')
    ax.quiver(*translation, *R_local_to_world[:, 1], color='g', length=1, normalize=True, label='Local yvec')
    ax.quiver(*translation, *R_local_to_world[:, 2], color='b', length=1, normalize=True, label='Local zvec')

    ax.scatter(*translation, color='k', s=60)
    ax.text(*translation, ' Camera Pos', color='k')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera View Frame')
    ax.legend()
    plt.draw()

def compute_translation(radius, angle_deg, z_height):
    angle_rad = np.deg2rad(angle_deg)
    x = radius * np.sin(angle_rad)
    y = radius * np.cos(angle_rad)
    return np.array([x, y, z_height])

# Thread-safe control variables
state = {"angle": angle_deg, "z": z_height, "radius": radius, "exit": False}

def on_press(key):
    try:
        if key == keyboard.Key.left:
            state["angle"] = (state["angle"] + angle_step) % 360  # Swapped direction
        elif key == keyboard.Key.right:
            state["angle"] = (state["angle"] - angle_step) % 360  # Swapped direction
        elif key == keyboard.Key.up:
            state["radius"] = max(radius_min, state["radius"] - radius_step)
        elif key == keyboard.Key.down:
            state["radius"] = min(radius_max, state["radius"] + radius_step)
        elif hasattr(key, 'char') and key.char:
            if key.char.lower() == 'w':
                state["z"] = min(z_max, state["z"] + height_step)
            elif key.char.lower() == 'z':
                state["z"] = max(z_min, state["z"] - height_step)
        elif key == keyboard.Key.esc:
            state["exit"] = True
            return False
    except Exception as e:
        print("Key handling error:", e)

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
######################################################################

# Start key listener in a separate thread
listener_thread = threading.Thread(target=start_listener)
listener_thread.start()

# Setup plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Main loop
while not state["exit"]:
    translation = compute_translation(state["radius"], state["angle"], state["z"])
    R_local_to_world = compute_frame(translation)
    update_plot(ax, translation, R_local_to_world)
    print("\nCamera Position (translation):", translation)
    print("R_local_to_world:\n", R_local_to_world)

    plt.pause(0.1)


plt.ioff()
plt.close(fig)

R= R_local_to_world
T=translation.reshape(3,1)



print("R:\n", R)
print("T:\n", T)
c2w = np.hstack((R, T))





nerf_path="./models/8bit_nerf.pth"
test_nerf(nerf_path,c2w)

