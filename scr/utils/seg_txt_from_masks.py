import cv2
import shutil
from pathlib import Path
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import screeninfo

def seg_txt_from_masks(path_to_images_folder: str,
                       path_to_masks_folder: str,
                       path_to_save_seg_labels: str) -> None:

    path_to_seg_labels = Path(path_to_save_seg_labels)

    if path_to_seg_labels.is_dir():
        shutil.rmtree(path_to_seg_labels)
    path_to_seg_labels.mkdir(parents=True, exist_ok=True)

    image_path = Path(path_to_images_folder)
    image_path_list = [image for image in sorted(image_path.rglob('*.jpg'), key=lambda x: int(x.stem.split('_')[-1]))]

    mask_path = Path(path_to_masks_folder)
    mask_path_list = [mask for mask in sorted(mask_path.rglob('*.jpg'), key=lambda x: int(x.stem.split('_')[-1]))]

    h, w = screeninfo.get_monitors()[0].height, screeninfo.get_monitors()[0].width
    
    
    for idx, mask in enumerate(mask_path_list):
        mask = cv2.imread(str(mask), 0)

        # Add a rectangle of 1 pixel width for the case of obj not fitting to the camera 
        mask_padded = cv2.rectangle(mask, (0, 0), (3072, 2048), (0, 0, 0) , 1) 

        # Threshold LAR
        # _, thresh_lar_255  = cv2.threshold(mask_padded, 250, 255, 0)

        # Threshold marker_250
        _, thresh1  = cv2.threshold(mask_padded, 240, 255, 0)
        _, thresh2  = cv2.threshold(mask_padded, 253, 255, 0)
        thresh_panel_250 = thresh1 - thresh2

        # Threshold panels_100
        _, thresh1  = cv2.threshold(mask_padded, 99, 255, 0)
        _, thresh2  = cv2.threshold(mask_padded, 101, 255, 0)
        thresh_panel_100 = thresh1 - thresh2 
        # Threshold panel_main_body_180
        _, thresh1  = cv2.threshold(mask_padded, 175, 255, 0)
        _, thresh2  = cv2.threshold(mask_padded, 185, 255, 0)
        thresh_main_body_180 = thresh1 - thresh2

        # Calculate contours for the panel
        # contours_255, _ = cv2.findContours(thresh_lar_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_250, _ = cv2.findContours(thresh_panel_250, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_100, _ = cv2.findContours(thresh_panel_100, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_180, _ = cv2.findContours(thresh_main_body_180, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
        # Make sure you consider only the largest contours
        contours_list = [contours_250,  contours_180, contours_100] # contours_255,
        largest_contours = []
        for cont in contours_list:
            if cont == ():
                continue
            else:
                largest_contour = max(cont, key=cv2.contourArea)
                largest_contours.append(largest_contour)
            

        # Iterate over the contours to form the segment txt file
        for j, contours in enumerate(largest_contours):
            polygon = []
            for i, point in enumerate(contours):
            
                x, y = point[0]
                polygon.append(round(x/3072, 2))
                polygon.append(round(y/2048, 2))
            
            if j == 0:
                with open(f'{path_to_seg_labels/image_path_list[idx].stem}.txt', 'w') as f:
                
                    # Iterate over the polygon (list of x, y coordinates)
                    for k, pnt in enumerate(polygon):
                    
                        if k == len(polygon) -1:
                            f.write(f'{pnt}\n')

                        elif k == 0:
                            f.write(f'{j} {pnt} ')
                        else:
                            f.write(f'{pnt} ')
            else:
                with open(f'{path_to_seg_labels/image_path_list[idx].stem}.txt', 'a') as f:

                    # Iterate over the polygon (list of x, y coordinates)
                    for i, pnt in enumerate(polygon):
                    
                        if i == len(polygon) -1:
                            f.write(f'{pnt}\n')

                        elif i == 0:
                            f.write(f'{j} {pnt} ')
                        else:
                            f.write(f'{pnt} ')


if __name__ == "__main__":
    path_to_images_folder = "../../data_3072px/images"  # Replace with your path
    path_to_masks_folder = "../../data_3072px/masks"  # Replace with your path
    path_to_save_seg_labels = "../../data_3072px/labels/seg_yolo"  # Replace with your path

    seg_txt_from_masks(path_to_images_folder, path_to_masks_folder, path_to_save_seg_labels)