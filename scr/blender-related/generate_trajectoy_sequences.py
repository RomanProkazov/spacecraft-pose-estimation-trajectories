import bpy
import starfish
import random
from pathlib import Path
from pathlib import Path
import numpy as np
from mathutils import Euler


def generate_sin_positions(nr_frames, amp_x=1.5, freq_x=7, 
                                    amp_y=-1.5, freq_y=1, 
                                    amp_z=0.0, freq_z=1, 
                                    end_pos=(0, 0, 0.0)):  
    t = np.linspace(0, np.pi, nr_frames)  # Time parameter from 0 to π
    
    # Oscillations followed by a smooth decay to 0
    x_values = amp_x * np.sin(freq_x * t) * (1 - t / np.pi) + end_pos[0]
    y_values = amp_y * np.sin(freq_y * t) * (1 - t / np.pi) + end_pos[1]
    z_values = amp_z * np.sin(freq_z * t) * (1 - t / np.pi) + end_pos[2]

    return list(zip(x_values, y_values, z_values))


def generate_sin_rotations(nr_frames,
                                        pitch_amp=50, pitch_freq=7,
                                        roll_amp=0, roll_freq=3,
                                        yaw_amp=0, yaw_freq=1,
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
        amp_x = random.uniform(-1.5, 1.5) 
        amp_y = random.uniform(-1.5, 1.5)
        x_freq = random.choice([1, 2, 3, 4, 5, 6, 7])
        y_freq = random.choice([1, 2, 3, 4, 5, 6, 7])
        
        pitch_amp = random.uniform(-50, 50)
        roll_amp = random.uniform(-50, 50)  
        yaw_amp = random.uniform(-50, 50)
        
        pitch_freq = random.choice([1, 2, 3, 4, 5, 6, 7])
        roll_freq = random.choice([1, 2, 3, 4, 5, 6, 7]) 
        yaw_freq = random.choice([1, 2, 3, 4, 5, 6, 7]) 

        trajectory_params.append((amp_x, amp_y, x_freq, y_freq, pitch_amp, roll_amp, yaw_amp,
                                    pitch_freq, roll_freq, yaw_freq))
    return trajectory_params



#-------------------MAIN SCRIPT---------------------#

abs_path = Path(bpy.path.abspath('//'))
data_path = abs_path / 'data' 
images_path = data_path / 'images'
labels_path = data_path / 'labels'

num_traj = 30
num_frames = 100
start_dist = 2.0
end_dist = 0.2
  

# Define Blender scene and objects
scene = bpy.data.scenes['Scene']
spacecraft = bpy.data.objects['ASSIEME']
sun = bpy.data.objects['Sun']
camera = bpy.data.objects['Camera']

# Clear previous animation
all_objects = [scene, spacecraft, sun, camera]
for obj in all_objects:
    obj.animation_data_clear()


trajectories = generate_trajectory_params(num_traj)
distances = np.linspace(start_dist, end_dist, num_frames)
initial_rotation = spacecraft.rotation_quaternion.copy()

# Open the JSON file once and write the opening bracket.
labels_file = labels_path / 'meta_keypoints.json'
with open(labels_file, 'w') as f:
    f.write("[")

total_frames = 0
for i, tr_pr in enumerate(trajectories):
    positions = generate_sin_positions(num_frames, amp_x=tr_pr[0], amp_y=tr_pr[1],
                                        freq_x=tr_pr[2], freq_y=tr_pr[3])
    print(f"trajectory {i}: {trajectories[i]}")
    # Corrected assignment operator here:
    rotations = generate_sin_rotations(num_frames, pitch_amp=tr_pr[4], roll_amp=tr_pr[5],
                                        yaw_amp=tr_pr[6], pitch_freq=tr_pr[7],
                                        roll_freq=tr_pr[8],yaw_freq=tr_pr[9])
    sequence = starfish.Sequence.standard(position=positions, pose=rotations, distance=distances)
    
    for j, frame in enumerate(sequence):
        frame_index = total_frames + j + 1  # Global frame number

        # Apply transformations
        spacecraft.location = frame.position
        spacecraft.rotation_quaternion = initial_rotation @ frame.pose
        camera.location = (0, 0, frame.distance)  # Camera moves along Z-axis
        
        # Insert keyframes using the global frame index
        scene.keyframe_insert(data_path="cycles.film_exposure", frame=frame_index)
        spacecraft.keyframe_insert('location', frame=frame_index)
        spacecraft.keyframe_insert('rotation_quaternion', frame=frame_index)
        camera.keyframe_insert('location', frame=frame_index)
        camera.keyframe_insert('rotation_quaternion', frame=frame_index)
        
        # Append frame data to the JSON file
        with open(labels_file, 'a') as f:
            # For the very first frame, do not prepend a comma.
            if total_frames == 0 and j == 0:
                f.write(frame.dumps())
            else:
                f.write(",\n" + frame.dumps())
    
    total_frames += len(sequence)

# Close the JSON array
with open(labels_file, 'a') as f:
    f.write("]")