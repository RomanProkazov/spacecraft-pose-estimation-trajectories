import bpy
import starfish
import mathutils
import math
import random
from starfish import utils
from pathlib import Path
from shutil import rmtree
from starfish.annotation import generate_keypoints
from starfish.annotation import project_keypoints_onto_image 
from starfish import Frame
from pathlib import Path
import json
import numpy as np
from mathutils import Quaternion, Euler


def generate_positions(nr_frames, start_pos=(-0.5, 0., 0.0), end_pos=(0, 0, 0.0)):
    x_values = np.linspace(start_pos[0], end_pos[0], nr_frames)  # X remains mostly constant
    y_values = np.linspace(start_pos[1], end_pos[1], nr_frames)  # Y remains mostly constant
    z_values = np.linspace(start_pos[2], end_pos[2], nr_frames)  # Gradual descent in Z
    return list(zip(x_values, y_values, z_values))  # List of (x, y, z) positions

def generate_sin_positions(nr_frames, amp_x=0.5, freq_x=3, 
                                    amp_y=0.0, freq_y=3, 
                                    amp_z=0.0, freq_z=1, 
                                    end_pos=(0, 0, 0.0)):
    t = np.linspace(0, np.pi, nr_frames)  # Time parameter from 0 to π
    
    # Oscillations followed by a smooth decay to 0
    x_values = amp_x * np.sin(freq_x * t) * (1 - t / np.pi) + end_pos[0]
    y_values = amp_y * np.sin(freq_y * t) * (1 - t / np.pi) + end_pos[1]
    z_values = amp_z * np.sin(freq_z * t) * (1 - t / np.pi) + end_pos[2]

    return list(zip(x_values, y_values, z_values))

    

def random_angles(min_angle, max_angle, num_frames):
    return [random.uniform(np.radians(min_angle), np.radians(max_angle)) for _ in range(num_frames)]


# Gnerate simple arrays of angles sequences from defined ones to zero
def generate_euler_rotations_simple(nr_frames, pitch_angle, roll_angle, yaw_angle):
    pitch = np.linspace(np.radians(pitch_angle), 0, nr_frames)
    roll = np.linspace(np.radians(roll_angle), 0, nr_frames) # Corrected line
    yaw = np.linspace(np.radians(yaw_angle), 0, nr_frames)
    quaternions = [Euler((p, y, r), 'XYZ').to_quaternion() for p, y, r in zip(pitch, yaw, roll)]
    return quaternions


def generate_sin_rotations(nr_frames,
                                        pitch_amp=15, pitch_freq=3,
                                        roll_amp=15, roll_freq=3,
                                        yaw_amp=-15, yaw_freq=1,
                                        end_angles=(0, 0, 0)):
 
    # Create normalized time values from 0 to 1
    t = np.linspace(0, 1, nr_frames)
    # Use a cosine-based decay: starts at 1 and decays to 0.
    decay = 0.5 * (1 + np.cos(t * np.pi))
    
    # Compute oscillatory angles (in degrees) with sinusoidal behavior and decay
    pitch = pitch_amp * np.sin(2 * np.pi * pitch_freq * t) * decay
    roll  = roll_amp  * np.sin(2 * np.pi * roll_freq  * t) * decay
    yaw   = yaw_amp * np.sin(2 * np.pi * yaw_freq * t) * decay
    
    # Convert to radians
    pitch_rad = np.radians(pitch)
    roll_rad  = np.radians(roll)
    yaw_rad   = np.radians(yaw)
    
    # For the final frame, force the orientation to end_angles (converted to radians)
    final_pitch, final_yaw, final_roll = np.radians(end_angles)
    
    quaternions = []
    for i in range(nr_frames):
        if i == nr_frames - 1:
            # Force final frame to be exactly the final orientation
            euler = Euler((final_pitch, final_yaw, final_roll), 'XYZ')
        else:
            # Use the generated angles; here we assume the desired order is (pitch, yaw, roll)
            euler = Euler((pitch_rad[i], yaw_rad[i], roll_rad[i]), 'XYZ')
        quaternions.append(euler.to_quaternion())
        
    return quaternions


def generate_trajectory_params(num_trajectories):
    trajectory_params = []
    for _ in range(num_trajectories):
        x_off = random.uniform(-0.5, 0.5)  # Lateral displacement in X
        y_off = random.uniform(-0.5, 0.5)  # Lateral displacement in Y
        yaw_ang = random.uniform(-20, 20)  # Yaw angle variation
        pitch_ang = random.uniform(-20, 20)  # Pitch angle variation
        roll_ang = random.uniform(-20, 20)  # Roll angle variation
        trajectory_params.append((x_off, y_off, yaw_ang, pitch_ang, roll_ang))
    return trajectory_params



#-------------------MAIN SCRIPT---------------------#

abs_path = Path(bpy.path.abspath('//'))
data_path = abs_path / 'data' 
images_path = data_path / 'images'
labels_path = data_path / 'labels'

start_dist = 2.0
end_dist = 0.2
nr_frames = 100  

pitch_angle = 15
roll_angle = 15
yaw_angle = 15

positions = generate_sin_positions(nr_frames)
rotations = generate_euler_rotations_simple(nr_frames, pitch_angle, roll_angle, yaw_angle)
rotations = generate_sin_rotations(nr_frames)
distances = np.linspace(start_dist, end_dist, nr_frames)

sequence = starfish.Sequence.standard(position=positions, pose=rotations, distance=distances)

# Define Blender scene and objects
scene = bpy.data.scenes['Scene']
spacecraft = bpy.data.objects['ASSIEME']
sun = bpy.data.objects['Sun']
camera = bpy.data.objects['Camera']

# Clear previous animation
all_objects = [scene, spacecraft, sun, camera]
for obj in all_objects:
    obj.animation_data_clear()
    

initial_rotation = spacecraft.rotation_quaternion.copy()
for i, frame in enumerate(sequence):
#    frame.setup(scene, spacecraft, camera, sun)
    spacecraft.location = frame.position
    spacecraft.rotation_quaternion = frame.pose
    
    spacecraft.rotation_quaternion = initial_rotation @ frame.pose

    # Set camera position
    # Camera moves only along the Z-axis to maintain distance
    camera.location = (0, 0, frame.distance)
    
    scene.keyframe_insert(data_path="cycles.film_exposure",frame=i+1)
    spacecraft.keyframe_insert('location', frame=i+1)
    spacecraft.keyframe_insert('rotation_quaternion', frame=i+1)
    camera.keyframe_insert('location', frame=i+1)
    camera.keyframe_insert('rotation_quaternion', frame=i+1)
    
    # Write data into a json file
    if i == 0:
        with open(labels_path / 'meta_keypoints.json', 'w') as f:
            f.write(f'[{frame.dumps()},\n')
                      
    elif i == len(sequence) - 1:
        with open(labels_path / 'meta_keypoints.json', 'a') as f:
            f.write(f'{frame.dumps()}]')
    else:
        with open(labels_path / 'meta_keypoints.json', 'a') as f:
            f.write(f'{frame.dumps()},\n')