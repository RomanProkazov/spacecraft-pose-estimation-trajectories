#!/bin/bash

# Change to Blender installation directory (optional, usually not needed)
cd /home/roman/opt/blender/blender-4.3.2-linux-x64 

# Run Blender in background mode (-b) with your .blend file and render frames
blender -b "/home/roman/Desktop/LUXEMBOURG PROJECT/blender-related/files/lux_sat_demo_1.blend1" -s 1 -e 3 -a -E CYCLES


# Change back to home directory (optional)
cd /home/roman
