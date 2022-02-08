import bpy
import json
import bmesh
from mathutils import Vector

filepath = "E:\\innvosion2022\\1_18_2022\\tpose-exercise\\tpose-exercise_world_landmark_result.json"

# Create sphere markers in Blender


def createMarker(name, location):
    scn = bpy.context.scene
    objName = name

    # Create an empty mesh and the object.
    mesh = bpy.data.meshes.new(objName)
    sphere = bpy.data.objects.new(objName, mesh)
    scn.collection.objects.link(sphere)

    # Construct the bmesh sphere and assign it to the blender mesh.
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, diameter=0.02)
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects[objName]
    ob.show_in_front = True

    # Create materials(color for the sphere)
    mat = bpy.data.materials.get("markers_sphere")
    if mat is None:
        mat = bpy.data.materials.new(name="markers_sphere")
        mat.diffuse_color = (1, 0, 0, 1)

    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)

    # Position the sphere into the correct location
    ob.location = location
    ob.scale = (0.5, 0.5, 0.5)
    ob.show_name = True

    return ob


# Create mid hip and mid shoulder, which are the extra joints that aren't existed in MediaPipe
def create_mid_joint(obj_list):
    num = len(obj_list)

    sum_loc = Vector((0, 0, 0))

    for obj in obj_list:
        sum_loc += obj.matrix_world.translation

    ave_loc = sum_loc/num
    return ave_loc


objs = bpy.data.objects

with open(filepath, 'r') as f:
    data = json.load(f)

current_frame = 1
bpy.context.scene.frame_set(current_frame)


landmark_names = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
                  'right_wrist', 'left_pinky_1', 'right_pinky_1', 'left_index_1', 'right_index_1', 'left_thumb_2', 'right_thumb_2', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']

landmark_names_2 = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
                    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot_index', 'right_foot_index']


for ele in data:
    # Set the number of frame
    # if current_frame > 10:
    # break

    if ele["frame"] != current_frame:

        mid_shoulder_loc = create_mid_joint(
            [objs["left_shoulder"], objs["right_shoulder"]])
        if objs.find("mid_shoulder") == -1:
            obj = createMarker("mid_shoulder", (0, 0, 0))
        else:
            obj = objs["mid_shoulder"]
        obj.location = mid_shoulder_loc
        bpy.context.view_layer.update()

        obj.keyframe_insert("location")
        obj.keyframe_insert("hide_viewport")

        mid_hip_loc = create_mid_joint([objs["left_hip"], objs["right_hip"]])

        if objs.find("mid_hip") == -1:
            obj = createMarker("mid_hip", (0, 0, 0))
        else:
            obj = objs["mid_hip"]
        obj.location = mid_hip_loc
        bpy.context.view_layer.update()

        obj.keyframe_insert("location")
        obj.keyframe_insert("hide_viewport")

        current_frame += 1
        bpy.context.scene.frame_set(current_frame)
        if current_frame > bpy.context.scene.frame_end:
            bpy.context.scene.frame_end = current_frame

    obj_name = landmark_names[(ele["keypoint_num"])]

    if obj_name not in landmark_names_2:
        continue

    if objs.find(obj_name) == -1:
        obj = createMarker(obj_name, (0, 0, 0))
    else:
        obj = objs[obj_name]

    obj.location = (ele["X"], ele["Y"], ele["Z"])
    bpy.context.view_layer.update()

    # You can hide the markers if their visibility is less than and equal to 0.5
    # False assigned in both case means all the markers are not hidden any more.
    if ele["Visibility"] > 0.5:
        obj.hide_viewport = False
    else:
        obj.hide_viewport = False
    obj.keyframe_insert("location")

# Insert keyframe for hidding action of the markers
# Change the second statement in the IF above and uncomment the new line, run it to see the result.
# It is hard to delete hidden objects from the View3D region. You can use the outline region to help you.

#    obj.keyframe_insert("hide_viewport")
