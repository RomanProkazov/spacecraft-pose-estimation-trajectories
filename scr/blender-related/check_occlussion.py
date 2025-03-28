import bpy
import mathutils
import bmesh  # Import bmesh for mesh conversion

def is_keypoint_visible(spacecraft, camera, keypoint_world, threshold=0.001):
    """
    Determines whether a keypoint is visible from the camera using ray casting.
    
    Args:
      spacecraft (Object): The mesh object of the spacecraft.
      camera (Object): The camera object.
      keypoint_world (mathutils.Vector): The 3D position of the keypoint in world space.
      threshold (float): A small tolerance value to avoid precision issues.
      
    Returns:
      bool: True if the keypoint is visible, False if it is occluded.
    """
    # Get camera location in world space
    cam_loc = camera.matrix_world.translation
    
    # Direction from camera to the keypoint
    direction = keypoint_world - cam_loc
    distance_to_keypoint = direction.length
    if distance_to_keypoint < threshold:
        # Keypoint is too close; assume it's visible.
        return True
    direction.normalize()

    # Get evaluated mesh for the spacecraft (for accurate ray casting)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = spacecraft.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    # Convert the mesh to a BMesh object
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Build the BVH tree from the BMesh
    bvh_tree = mathutils.bvhtree.BVHTree.FromBMesh(bm)

    # Free the BMesh and evaluated mesh to avoid memory leaks
    bm.free()
    eval_obj.to_mesh_clear()
    
    # Cast a ray from the camera toward the keypoint
    location, normal, face_index, hit_distance = bvh_tree.ray_cast(cam_loc, direction)
    
    # If an intersection was found and it is closer than the keypoint (with tolerance), then it's occluded
    if location is not None and hit_distance < (distance_to_keypoint - threshold):
        return False
    else:
        return True


def get_keypoints_world_positions(spacecraft, sat_model):
    """
    Transforms the default keypoints of the spacecraft into their world-space positions.
    
    Args:
      spacecraft (Object): The spacecraft object in the Blender scene.
      sat_model (list of mathutils.Vector): The default keypoints in the spacecraft's local space.
      
    Returns:
      list of mathutils.Vector: The keypoints transformed into world space.
    """
    world_keypoints = []
    for kp in sat_model:
        # Transform the keypoint from local space to world space
        world_kp = spacecraft.matrix_world @ kp
        world_keypoints.append(world_kp)
    return world_keypoints


# Example usage:
# Assume we have a list of keypoints in local coordinates (as mathutils.Vector objects)
sat_model = [
    mathutils.Vector((-0.04, -0.04, 0)),
    mathutils.Vector((-0.04, 0.04, 0)),
    mathutils.Vector((0.04, 0.04, 0)),
    mathutils.Vector((0.04, -0.04, 0)),

    mathutils.Vector((-0.15, -0.15, -0.001984)),
    mathutils.Vector((-0.15, 0.15, -0.001984)),
    mathutils.Vector((0.15, 0.15, -0.001984)),
    mathutils.Vector((0.15, -0.15, -0.001984)),

    mathutils.Vector((-0.15, -0.15, -0.301984)),
    mathutils.Vector((-0.15, 0.15, -0.301984)),
    mathutils.Vector((0.15, 0.15, -0.301984)),
    mathutils.Vector((0.15, -0.15, -0.301984)),

    mathutils.Vector((-0.6499, -0.153, -0.14724)),
    mathutils.Vector((-0.6499, 0.15, -0.14724)),
    mathutils.Vector((0.6499, 0.15, -0.14724)),
    mathutils.Vector((0.6499, -0.153, -0.14723))
]

spacecraft = bpy.data.objects['ASSIEME']
camera = bpy.data.objects['Camera']

# Get the keypoints in world space for the current frame
world_keypoints = get_keypoints_world_positions(spacecraft, sat_model)

# Check visibility for each keypoint and print the result
for idx, kp in enumerate(world_keypoints):
    vis = is_keypoint_visible(spacecraft, camera, kp)
    print(f"Keypoint {idx}: {'Visible' if vis else 'Occluded'}")
