#!/bin/bash


# Run Blender in background mode (-b) with your .blend file and render frames
# blender -b "/home/roman/Desktop/LUXEMBOURG PROJECT/blender-related/files/Working_trajectories_with_offset.blend" -s 1 -e 5000 -a -E CYCLES
blender -b "/home/roman/Desktop/FLOATING PLATFORM PROJECT/blender-related/floating_platform.blend" -s 1 -e 5000 -a -E CYCLES
