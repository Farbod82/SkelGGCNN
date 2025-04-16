import numpy as np
import cv2
from utility.utility import remove_objs_from_sim, get_all_mesh_names, get_rgb_img, get_depth_img,load_object
import time
import os
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import random
import logging
import argparse






class DatasetCollector():
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.camera_names = ["cam1","cam2","cam3","cam4","cam5"]
        self.light_name = "Spotlight" 
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.sim.stopSimulation()
        self.sim.setLightParameters(self.sim.getObject('/DefaultLightA'), 1)
        self.sim.setLightParameters(self.sim.getObject('/DefaultLightB'), 1)  
        self.sim.setLightParameters(self.sim.getObject('/DefaultLightC'), 1)
        
    def set_light_on(self):
        handle = self.sim.getObject(f"/{self.light_name}")
        self.sim.setLightParameters(handle, 1, None, None, None)
        time.sleep(0.3)
        

    def set_light_off(self):
        handle = self.sim.getObject(f"/{self.light_name}")
        self.sim.setLightParameters(handle, 0, None, None, None)
        time.sleep(0.3)


    def capture_image_data(self):
        rgb_images = []
        depth_images = []
        masks = []
        for cam in self.camera_names:
            cam_handle = self.sim.getObject(f'/{cam}')
            rgb_image = get_rgb_img(cam_handle,self.sim)
            depth_image = get_depth_img(cam_handle,self.sim)
            mask = self.capture_mask(cam_handle)
            
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            masks.append(mask)
        return rgb_images, depth_images, masks
    
    def capture_mask(self,cam_handle):        
        self.set_light_off()
        rgb_image = get_rgb_img(cam_handle,self.sim)
        color = rgb_image[0,0]
        lower = np.array(color) - 3
        upper = np.array(color) + 3
        mask = cv2.inRange(rgb_image,lower,upper)
        self.set_light_on()
        return mask
    
    def save_data(self,rgb_images,depth_images, masks,save_path,mesh_name):
        path = os.path.join(save_path,mesh_name)
        os.makedirs(path, exist_ok=True)
        rgb_path = os.path.join(path,"rgb")
        depth_path = os.path.join(path,"depth")
        mask_path = os.path.join(path,"mask")
        
        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(mask_path, exist_ok=True)

        for i in range(len(rgb_images)):
            cv2.imwrite(os.path.join(rgb_path,f"{i+1}.png"),rgb_images[i])
            np.save(os.path.join(depth_path,f"{i+1}.npy"),depth_images[i])
            cv2.imwrite(os.path.join(mask_path,f"{i+1}.png"),masks[i])
            
    def collect_data(self,mesh_path,save_path):
    
        remove_objs_from_sim(self.sim)
        
        mesh_names = get_all_mesh_names(mesh_path)
        for mesh in mesh_names:
            obj_handle = load_object(os.path.join(mesh_path,mesh), self.client, self.sim)[0]
            self.sim.startSimulation()
            time.sleep(0.5)
            self.sim.setObjectPosition(obj_handle,[0,0,0], -1)
            rgb_images, depth_images, masks = self.capture_image_data()
            self.save_data(rgb_images,depth_images,masks,save_path,mesh)
            logging.info(f"Data capture completed for mesh: {mesh}")
            remove_objs_from_sim(self.sim)
            self.sim.stopSimulation()
            time.sleep(0.4)


def parse_args():
    parser = argparse.ArgumentParser(description='Dataset Collection for Grasping')
    parser.add_argument('--mesh-path', type=str, required=True, help='Path to the folder containing mesh files')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save the collected dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generator = DatasetCollector()
    generator.collect_data(args.mesh_path, args.save_path)

