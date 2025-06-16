import bpy
import starfish
import random
from pathlib import Path
from pathlib import Path
import numpy as np
from mathutils import Vector, Euler
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
        ox_amp = random.uniform(0.0, 0.2)  # Offset amplitude in normalized (0,1) space
        oy_amp = random.uniform(0.0, 0.2)  
        ox_freq = random.choice([1, 2])
        oy_freq = random.choice([1, 2])
        ox_amp =  0.2  # Offset amplitude in normalized (0,1) space
        oy_amp =   0.2     
#        ox_freq = 2
#        oy_freq = 2  
        pitch_amp = random.uniform(-15 , 15)  
        roll_amp = random.uniform(-15, 15)  
        yaw_amp = random.uniform(-15, 15) 
         
        pitch_freq = random.choice([1, 2, 3])
        roll_freq = random.choice([1, 2, 3 ]) 
        yaw_freq = random.choice([1, 2, 3]) 
        
        trajectory_params.append((ox_amp, oy_amp, ox_freq, oy_freq, pitch_amp, roll_amp, yaw_amp, pitch_freq, roll_freq, yaw_freq))
    return trajectory_params


# ------ EARTH ROTATION SETUP ------
def setup_earth_rotation(total_frames, rot_angle_deg):
    earth = bpy.data.objects.get('Earth_all')
    if not earth:
        print("Warning: Earth object not found - rotation won't be animated")
        return

    # Set axial tilt (23.4° on X axis)
    earth.rotation_euler = (0,  0, 0) 
    
    # Clear existing animationSSS   
    
    # Set keyframes for full rotation
    earth.keyframe_insert(data_path='rotation_euler', frame=1)
    earth.rotation_euler.y = math.radians(rot_angle_deg)  # Full rotation on Z axis
    earth.keyframe_insert(data_path='rotation_euler', frame=total_frames) 
    
    # Set linear interpolation for constant rotation
    if earth.animation_data and earth.animation_data.action:
        for fcurve in earth.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR' 




#------------------------------------------MAIN SCRIPT-------------------------------------------#
abs_path = Path(bpy.path.abspath('//'))
data_path = abs_path / 'data' 
images_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/images"
labels_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/labels/meta_keypoints.json"

num_traj = 20 
num_frames = 1000
start_dist = 30
end_dist = 0.71  
sun_rot_angle = 70
scene_scale = 400000
elevation_deg_list = np.linspace(-45, 70, num_traj)
azimuth_deg_list = np.linspace(0, 0, num_traj)

# Define Blender scene and objects
scene = bpy.data.scenes['Scene']
spacecraft = bpy.data.objects['skymed_body_axis']
sun = bpy.data.objects['Sun']
camera = bpy.data.objects['Camera']


# Clear previous animation
all_objects = [scene, spacecraft, sun, camera]
for obj in all_objects:
    obj.animation_data_clear()
    
# Initialize Earth rotation before main loop
setup_earth_rotation(num_frames * num_traj, sun_rot_angle*num_traj)  # Total frames across trajectories



trajectories = generate_trajectory_params(num_traj)
#distances = np.linspace(start_dist, end_dist, num_frames)
  # Match your pose estimation setup
distances = np.linspace(start_dist, end_dist, num_frames) / scene_scale

total_frames = 0
for i, tr_pr in enumerate(trajectories):
    offset_list = generate_sin_offsets(num_frames, amp_x=tr_pr[0], amp_y=tr_pr[1],
                                        freq_x=tr_pr[2], freq_y=tr_pr[3])
    # Corrected assignment operator here:
    rotations = generate_sin_rotations(num_frames, pitch_amp=tr_pr[4], roll_amp=tr_pr[5],
                                        yaw_amp=tr_pr[6], pitch_freq=tr_pr[7],
                                        roll_freq=tr_pr[8],yaw_freq=tr_pr[9])
    sequence = starfish.Sequence.standard(pose=rotations, distance=distances, offset=offset_list)
    # Calculate start frame for this trajectory
    trajectory_start_frame = i * num_frames + 1
    
    # --- SUN DIRECTION PER TRAJECTORY ---
    sun = bpy.data.objects['Sun']
    sun.rotation_mode = 'QUATERNION'
    
    # Get angles for this trajectory
    azimuth_deg = azimuth_deg_list[i]
    elevation_deg = elevation_deg_list[i]
    
    # Convert to direction vector
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)
    x = math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = math.sin(elevation_rad)
    direction = Vector((x, y, z)).normalized()
    
    # Set sun rotation and location
    sun.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    sun.location = -direction * 1000  # Keep sun far away
    
    # Keyframe sun properties at trajectory start
    sun.keyframe_insert(data_path='rotation_quaternion', frame=trajectory_start_frame)
    sun.keyframe_insert(data_path='location', frame=trajectory_start_frame)
    
    # Force abrupt change between trajectories
    if sun.animation_data and sun.animation_data.action:
        for fcurve in sun.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                if kp.co.x == trajectory_start_frame:
                    kp.interpolation = 'CONSTANT'
    


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
        camera.keyframe_insert('rotation_quaternion', frame=frame_index)
        
#        # Project keypoints and save them in the frame data [(y, x) - normalized]
#        frame.keypoints = project_keypoints_onto_image(sat_model, scene, spacecraft, camera)
        
        # Write data into a json file
        if frame_index == 1:
            with open(labels_path, 'w') as f:
                f.write(f'[{frame.dumps()},\n')
                
        elif frame_index == len(sequence)*len(trajectories):
            with open(labels_path, 'a') as f:
                f.write(f'{frame.dumps()}]')
        else:
            with open(labels_path, 'a') as f:
                f.write(f'{frame.dumps()},\n')
                
    total_frames += num_frames  
    
 