"""
Mimic ARM-NET imscore
"""
import cv2
import os
import tqdm
import numpy as np

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def get_imscore_from_path(image_path):
    image = cv2.imread(image_path)
    imscore = laplacian(image).mean()
    return imscore

def main():
    # initialize
    patch_dir = '../../data/PATCHES_RGB/8x8/Set14/image_SRF_2/'
    save_dir = './experiments/ANALYZE/EUNAF_EDSRx2_bl_x2_nb16_nf64_st2/'
    os.makedirs(save_dir, exist_ok=True)
    
    patch_paths = [p for p in os.listdir(patch_dir)]
    num_LR = len(patch_paths) // 2
    scale = 2
    
    imscores = list()
    
    for p in tqdm.tqdm(range(num_LR), total=num_LR):
        path = os.path.join(patch_dir, 'img_%03d_SRF_%d_LR.png' % (p+1,scale))
        imscore = get_imscore_from_path(path) 
        imscores.append(imscore)   
        
    np_save_file = os.path.join(save_dir, 'edge_Set14RGB.npy')
    np.save(np_save_file, np.array(imscores))
        
if __name__ == '__main__':
    main()