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
    patch_dir = '/mnt/disk1/nmduong/FusionNet/data/PATCHES_RGB/32x32/Test2K/HR/X4'
    save_dir = './experiments/ANALYZE/EUNAF_SRResNetxN_x4_nb16_nf64_st0/'
    os.makedirs(save_dir, exist_ok=True)
    
    patch_paths = [p for p in os.listdir(patch_dir)]
    num_LR = len(patch_paths)
    scale = 4
    
    imscores = list()
    
    for p in tqdm.tqdm(range(num_LR), total=num_LR):
        path = os.path.join(patch_dir, str(1201+p)+'.png')
        imscore = get_imscore_from_path(path) 
        imscores.append(imscore)   
        
    np_save_file = os.path.join(save_dir, 'edge_Test2K.npy')
    np.save(np_save_file, np.array(imscores))
        
if __name__ == '__main__':
    main()