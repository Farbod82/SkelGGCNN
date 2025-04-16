import os
import time
import math
import numpy as np



def load_object(file_path, client, sim):
    #Code written by Alireza Beigy (https://github.com/AliRezaBeigy)
    obj = sim.importShape(0, file_path, 0, 0, 1)

    if obj is None:
        return None

    return create_simulation_shapes(client, sim, obj, file_path)


def create_simulation_shapes(client, sim, obj, file_id):
    #Code written by Alireza Beigy (https://github.com/AliRezaBeigy)
    env_handle = sim.getObject('/env')
    shape_handles = []
    tries = 0
    while tries < 5:
        try:
            tries += 1
            sim.relocateShapeFrame(obj, [0, 0, 0, 0, 0, 0, 0])

            desired_height = 0.20
            max_width = 0.7

            shapeBB = sim.getShapeBB(obj)
            bb = shapeBB[0]

            current_height = bb[2]
            current_width = max(abs(bb[0]), abs(bb[1]))

            if current_height > current_width:
                sim.setObjectOrientation(obj, [0, math.pi / 2, 0], -1)

                shapeBB = sim.getShapeBB(obj)
                bb = shapeBB[0]
                current_height = bb[2]
                current_width = max(abs(bb[0]), abs(bb[1]))

            scale_factor_height = desired_height / current_height
            scale_factor_width = max_width / current_width if (current_width * scale_factor_height) > max_width else 1000

            scale_factor = min(scale_factor_height, scale_factor_width)

            scaled_bb = [dim * scale_factor for dim in bb]

            sim.setShapeBB(obj, scaled_bb)

            height = scaled_bb[2]
            sim.setObjectPosition(obj, [0, 0, height / 2 + 0.05], -1)

            sim.setShapeColor(obj, "", sim.colorcomponent_ambient, [0.5, 0.5, 0.5])
            sim.setObjectInt32Param(obj, sim.shapeintparam_respondable, 1)
            sim.setObjectInt32Param(obj, sim.shapeintparam_static, 0)
            sim.resetDynamicObject(obj)

            sim.setObjectAlias(obj, "object")
            sim.setObjectParent(obj, env_handle, True)

            shape_handles.append(obj)
            friction = 1.0
            sim.setFloatProperty(obj, 'mass', 0.1)
            sim.setBoolProperty(obj, 'bullet.stickyContact', True)
            sim.setBoolProperty(obj, 'bullet.autoShrinkConvexMeshes', True)
            sim.setBoolProperty(obj, 'bullet.customCollisionMarginEnabled', True)
            sim.setFloatProperty(obj, 'bullet.friction', friction)
            sim.setFloatProperty(obj, 'bullet.linearDamping', 0.2)
            sim.setFloatProperty(obj, 'bullet.angularDamping', 0.2)
            sim.setFloatProperty(obj, 'bullet.frictionOld', friction)
            sim.setFloatProperty(obj, 'newton.staticFriction', 3)
            sim.setFloatProperty(obj, 'newton.kineticFriction', 2.5)
            sim.setFloatProperty(obj, 'newton.restitution', 1)
            sim.setFloatProperty(obj, 'newton.linearDrag', 0.5)
            sim.setFloatProperty(obj, 'newton.angularDrag', 0.5)
            sim.setBoolProperty(obj, 'newton.fastMoving', False)
            sim.setFloatProperty(obj, 'bullet.customCollisionMarginValue', 0.05)
            sim.setFloatProperty(obj, 'bullet.customCollisionMarginConvexValue', 0.001)

            sim.setFloatProperty(sim.handle_scene, 'mujoco.impratio', 50)

            break
        except Exception as e:
            print(f"Failed to load: {file_id} due to {e}")
            time.sleep(0.1)
    return shape_handles





def get_depth_img(cam_handle,sim):
    depth, resolution = sim.getVisionSensorDepth(cam_handle, 1)
    depth = np.frombuffer(depth, np.float32)
    depth = depth.reshape((resolution[1], resolution[0], 1))

    return depth

def get_rgb_img(cam_handle,sim):
    rgb_image, resolution = sim.getVisionSensorImg(cam_handle)
        
    rgb_image = np.frombuffer(rgb_image, dtype=np.uint8)
    rgb_image = rgb_image.reshape((resolution[1], resolution[0], 3))
    return rgb_image    


def remove_objs_from_sim(sim):
    env_handle = sim.getObject('/env')
    obj_handles = sim.getObjectsInTree(env_handle, sim.handle_all, 0)
    obj_handles.remove(env_handle)
    if obj_handles:
        sim.removeObjects(obj_handles)


def get_all_mesh_names(path):
    return [f for f in os.listdir(path) if f.endswith('.obj')]
