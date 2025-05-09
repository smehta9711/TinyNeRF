import json
import numpy as np  # Add this line to import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_transforms_from_json(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # List to store translation values
    translations = []

    # Iterate over the frames in the JSON data and extract translations
    for frame in data.get("frames", []):
        matrix = frame.get("transform_matrix", [])
        
        # Extract translation from the last column (ignoring the last row)
        translation = matrix[0][3], matrix[1][3], matrix[2][3]  # (x, y, z)
        translations.append(translation)

    # Convert translations to numpy array for convenience
    translations = np.array(translations)

    # Plotting the top view (X-Y plane)
    plt.figure(figsize=(10, 6))
    plt.subplot(121)  # Top view: X-Y plane
    plt.scatter(translations[:, 0], translations[:, 1], c='r', label="Translations")
    plt.title("Top View (X-Y Plane)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    # Plotting the 3D view (X, Y, Z axes)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(122, projection='3d')  # 3D view
    ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], c='b', label="Translations")
    ax.set_title("3D View (X, Y, Z)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


plot_transforms_from_json('./transforms_test.json')

