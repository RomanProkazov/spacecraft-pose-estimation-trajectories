import bpy
import starfish
import random
from pathlib import Path
from pathlib import Path
import numpy as np
from mathutils import Euler
import math
import json


def generate_sin_offsets(nr_frames, amp_x=0.5, freq_x=3, 
                                    amp_y=0.5, freq_y=3, 
                                    end_offset=(0.5, 0.5)):  
    t = np.linspace(0, np.pi, nr_frames) 
    # Oscillations followed by a smooth decay to 0 (so it stabilizes at the end)
    ox_values = amp_x * np.sin(freq_x * t) * (1 - t / np.pi) + end_offset[0]
    oy_values = amp_y * np.sin(freq_y * t) * (1 - t / np.pi) + end_offset[1]
    return list(zip(ox_values, oy_values))



def generate_sin_rotations(nr_frames,
                           pitch_amp=50, pitch_freq=7,
                           roll_amp=0, roll_freq=3,
                           yaw_amp=0, yaw_freq=1,
                           end_angles=(0, 0, 0)):
    
    t = np.linspace(0, 1, nr_frames)
    decay = 0.5 * (1 + np.cos(t * np.pi))
  
    pitch = pitch_amp * np.sin(2 * np.pi * pitch_freq * t) * decay
    roll  = roll_amp  * np.sin(2 * np.pi * roll_freq  * t) * decay
    yaw   = yaw_amp * np.sin(2 * np.pi * yaw_freq * t) * decay
   
    pitch_rad = np.radians(pitch)
    roll_rad  = np.radians(roll)
    yaw_rad   = np.radians(yaw)
  
    final_pitch, final_yaw, final_roll = np.radians(end_angles)
    quaternions = []
    for i in range(nr_frames):
        if i == nr_frames - 1:
            # Force final frame to be exactly the final orientation
            euler = Euler((final_pitch, final_yaw, final_roll), 'XYZ')
        else:
            euler = Euler((pitch_rad[i], yaw_rad[i], roll_rad[i]), 'XYZ')
        quaternions.append(euler.to_quaternion())     
    return quaternions


def generate_trajectory_params(num_trajectories):
    trajectory_params = []
    for _ in range(num_trajectories):
        ox_amp = random.uniform(0.1, 0.4)  # Offset amplitude in normalized (0,1) space
        oy_amp = random.uniform(0.1, 0.4)  
        ox_freq = random.choice([1, 2, 3, 4, 5, 6, 7])
        oy_freq = random.choice([1, 2, 3, 4, 5, 6, 7])
        
        pitch_amp = random.uniform(-50, 50)
        roll_amp = random.uniform(-50, 50)  
        yaw_amp = random.uniform(-50, 50)
        
        pitch_freq = random.choice([1, 2, 3, 4, 5, 6, 7])
        roll_freq = random.choice([1, 2, 3, 4, 5, 6, 7]) 
        yaw_freq = random.choice([1, 2, 3, 4, 5, 6, 7]) 
        
        trajectory_params.append((ox_amp, oy_amp, ox_freq, oy_freq, pitch_amp, roll_amp, yaw_amp, pitch_freq, roll_freq, yaw_freq))
    return trajectory_params



abs_path = Path(bpy.path.abspath('//'))
data_path = abs_path / 'data' 
images_path = data_path / 'images'
labels_path = data_path / 'labels'

num_traj = 10
num_frames = 500
start_dist = 2.0
end_dist = 0.105
  

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


total_frames = 0
for i, tr_pr in enumerate(trajectories):
    offset_list = generate_sin_offsets(num_frames, amp_x=tr_pr[0], amp_y=tr_pr[1],
                                        freq_x=tr_pr[2], freq_y=tr_pr[3])
    print(f"trajectory {i}: {trajectories[i]}")
    # Corrected assignment operator here:
    rotations = generate_sin_rotations(num_frames, pitch_amp=tr_pr[4], roll_amp=tr_pr[5],
                                        yaw_amp=tr_pr[6], pitch_freq=tr_pr[7],
                                        roll_freq=tr_pr[8],yaw_freq=tr_pr[9])
    sequence = starfish.Sequence.standard(pose=rotations, distance=distances, offset=offset_list)

    # Iterate over the sequence (list of frames)
    for j, frame in enumerate(sequence):
        frame_index = total_frames + j + 1
        
        frame.setup(scene, spacecraft, camera, sun)
        # Insert keyframes using the global frame index
        scene.keyframe_insert(data_path="cycles.film_exposure", frame=frame_index)
        spacecraft.keyframe_insert('location', frame=frame_index)
        spacecraft.keyframe_insert('rotation_quaternion', frame=frame_index)
        camera.keyframe_insert('location', frame=frame_index)
        camera.keyframe_insert('rotation_quaternion', frame=frame_index)
        
        # Write data into a json file
        if frame_index == 1:
            with open(labels_path / 'meta_keypoints.json', 'w') as f:
                f.write(f'[{frame.dumps()},\n')
                
        elif frame_index == len(sequence)*len(trajectories):
            with open(labels_path / 'meta_keypoints.json', 'a') as f:
                f.write(f'{frame.dumps()}]')
        else:
            with open(labels_path / 'meta_keypoints.json', 'a') as f:
                f.write(f'{frame.dumps()},\n')
                
    total_frames += num_frames