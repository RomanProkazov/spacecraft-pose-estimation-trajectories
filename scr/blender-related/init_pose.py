import bpy
from starfish import Frame


# Define Blender scene 
scene = bpy.data.scenes['Scene']

# Define Spacecraft
spacecraft = bpy.data.objects['ASSIEME']

# Define Suns
sun = bpy.data.objects['Sun']

# Define Camera
camera = bpy.data.objects['Camera']

frame = Frame()

# Setup the frame
frame.setup(scene, spacecraft, camera, sun)