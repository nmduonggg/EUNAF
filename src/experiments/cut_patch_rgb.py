import numpy as np
import torch 
import os
import skimage.color as sc
import imageio
import tqdm
import argparse
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Set5', help='Set5, Set14')
    parser.add_argument('--root', type=str, default='../../data/')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='../../data/PATCHES_RGB/')
    parser.add_argument('--patch_size', type=int, default=8)

    args = parser.parse_args()
    
    return args 

def get_total_instances(set_name):
    if set_name=='Set5': return 5
    elif set_name=='Set14': return 14
    else: assert 0, 'not available dataset'

def get_patch(im_lr, im_hr, ix, iy, lr_patch_size, scale):
    lw = im_lr.shape[2]
    lh = im_lr.shape[1]

    im_lr_patch = im_lr[:, iy      : iy+lr_patch_size,        ix      : ix+lr_patch_size       ]
    im_hr_patch = im_hr[:, iy*scale:(iy+lr_patch_size)*scale, ix*scale:(ix+lr_patch_size)*scale]

    return im_lr_patch, im_hr_patch

def load_image_as_Tensor(im_file_name):
    data = imageio.imread(im_file_name)
    if data.ndim == 2:
        data = np.stack((data,)*3, axis=-1)

    data = np.ascontiguousarray(np.transpose(data, (2,0,1)))
    data = torch.Tensor(data)
    return data

def main():
    args = get_parser() 
    args.root = os.path.join(args.root, args.dataset, f'image_SRF_{args.scale}')
    args.save_dir = os.path.join(args.save_dir, f'{args.patch_size}x{args.patch_size}', args.dataset, f'image_SRF_{args.scale}')
    os.makedirs(args.save_dir, exist_ok=True)
    
    num_instances = get_total_instances(args.dataset)
    cnt = 0
    for i in tqdm.tqdm(range(num_instances), total=num_instances):
        im_fn = os.path.join(args.root, 'img_%03d_SRF_%d_LR.png' % (i+1,args.scale))
        x_data = load_image_as_Tensor(im_fn)
        
        gt_fn = os.path.join(args.root, 'img_%03d_SRF_%d_HR.png' % (i+1,args.scale))
        y_data = load_image_as_Tensor(gt_fn)
        
        imh, imw = x_data.shape[1:3]
        gth, gtw = y_data.shape[1:3]
        
        patches_h = imh // args.patch_size
        patches_w = imw // args.patch_size
        
        for ix in range(patches_w):
            for iy in range(patches_h):
                im_patch, gt_patch = get_patch(x_data, y_data, ix, iy, args.patch_size, args.scale)
                im_patch = im_patch.permute(1, 2, 0).numpy().astype(np.uint8)[:, :, :3]
                gt_patch = gt_patch.permute(1, 2, 0).numpy().astype(np.uint8)[:, :, :3]
                
                im_out_fn = os.path.join(args.save_dir, 'img_%03d_SRF_%d_LR.png' % (cnt+1,args.scale))
                gt_out_fn = os.path.join(args.save_dir, 'img_%03d_SRF_%d_HR.png' % (cnt+1,args.scale))
                imageio.v3.imwrite(im_out_fn, im_patch)
                imageio.v3.imwrite(gt_out_fn, gt_patch)
                
                cnt += 1
                
    print(f"[RESULT] {args.dataset} contains {cnt} patches of size {args.patch_size}")
                
if __name__ == '__main__':
    main()
                
"""
Set 14 - 8x8 - 12492 (x2)
Set 14 - 8x8 - 3069 (x4)
"""
        
        