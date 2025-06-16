import bpy
import math
from mathutils import Vector, Euler

# --- USER INPUT: angles in degrees (azimuth, elevation, roll) ---
# For example: 45° azimuth (horizontal), 30° elevation (up), 0° roll
azimuth_deg = 0 # Rotation around Z (horizontal, 0 = +X)
elevation_deg = 70            # Angle up from XY plane (0 = horizon, +90 = up) [0 -90]
roll_deg = 0       # Usually 0 for sun direction

# --- CONVERT ANGLES TO DIRECTION VECTOR ---
# Azimuth: 0° = +X, 90° = +Y
azimuth_rad = math.radians(azimuth_deg)
elevation_rad = math.radians(elevation_deg)

# Spherical to Cartesian conversion
x = math.cos(elevation_rad) * math.cos(azimuth_rad)
y = math.cos(elevation_rad) * math.sin(azimuth_rad)
z = math.sin(elevation_rad)
direction = Vector((x, y, z)).normalized()

# --- SET SUN ROTATION ---
sun = bpy.data.objects.get('Sun')
if sun is None:
    raise ValueError("No object named 'Sun' found.")

sun.rotation_mode = 'QUATERNION'
sun.rotation_quaternion = direction.to_track_quat('-Z', 'Y')

# (Optional) Move Sun far away for clarity (not required for lighting)
sun.location = -direction * 1000

print(f"Sun direction set to azimuth {azimuth_deg}°, elevation {elevation_deg}° (world space)")
