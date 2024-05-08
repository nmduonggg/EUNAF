import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
import matplotlib.pyplot as plt
import numpy as np

#custom modules
import data
import evaluation
import loss
import model as supernet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
from option import parser
from template import test_template as template


args = parser.parse_args()
args.infer_file = '../../data/nm.jpg'

if args.template is not None:
    template.set_template(args)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}' if args.n_resgroups > 0 \
        else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}'
    out_dir = os.path.join(args.cv_dir, fname)
    args.weight = os.path.join(out_dir, '_best.t7')
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight), strict=False)
core.cuda()

out_dir = os.path.join(args.analyze_dir, name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}', args.testset_tag)
print('Load to: ', out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
def gray2heatmap(image):
    heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    
    return heatmap
    
def process_unc_map(masks, to_heatmap=True, 
                    rescale=True, abs=True, 
                    amplify=False, scale_independent=False):
    """
    use for mask with value range [-1, inf]
    apply sigmoid and rescale
    """      
    masks = torch.stack(masks, dim=0)
        
    if abs:
        masks = torch.exp(masks)
    
    pmin = torch.min(masks)
    pmax = torch.max(masks)
    agg_mask = 0
    masks_numpy = []
    for i in range(len(masks)):
        
        if amplify:
            mask = masks[i,...].squeeze(0).permute(1,2,0).cpu().numpy()
            Q1 = np.percentile(mask, 25)
            Q3 = np.percentile(mask, 75)
            IQR = Q3 - Q1

            # Compute lower and upper bounds to filter outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Apply clipping to reduce the impact of outliers
            mask = np.clip(mask, lower_bound, upper_bound)
            
            if rescale:
                pmin = np.min(mask) 
                pmax = np.max(mask)  
                mask = (mask - pmin) / (pmax - pmin)
                # mask = (mask*255).round().astype(np.uint8)
        
        else:
            # mask = torch.abs(mask)
            mask = masks[i, ...]
            if scale_independent: 
                pmin = torch.min(mask)    
                pmax = torch.max(mask)
            
            # print(f'Mask {i}:', torch.mean(mask))
            if rescale:
                mask = (mask - pmin) / (pmax - pmin)
            
            mask = mask.squeeze(0).permute(1,2,0)
            agg_mask += mask
            mask = mask.cpu().numpy()
            
        if rescale:
            mask = (mask*255).round().astype(np.uint8) 
        if to_heatmap:
            mask = gray2heatmap(mask)  # gray -> bgr
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        masks_numpy.append(mask)
        
    return masks_numpy

def visualize_unc_map_binary(masks, id, val_perfs):
    new_out_dir = out_dir
    os.makedirs(new_out_dir, exist_ok=True)
    save_file = os.path.join(new_out_dir, f"img_{id}_mask_binary.jpeg")
    
    # masks_np = process_unc_map(masks, False, False, False)
    # masks_np_percentile = [(m > np.percentile(m, 90))*255 for m in masks_np]
    masks_np_percentile = process_unc_map(masks, to_heatmap=False, rescale=True, amplify=True, abs=True)
    
    fig, axs = plt.subplots(1, len(masks_np_percentile), 
                            tight_layout=True, figsize=(60, 20))
    for i, m in enumerate(masks_np_percentile):
        axs[i].imshow(m, cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'block {i} - perf {val_perfs[i].detach().item()}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()

def visualize_unc_map(masks, id, im=False):
    new_out_dir = out_dir
    os.makedirs(new_out_dir, exist_ok=True)
    save_file = os.path.join(new_out_dir, f"img_{id}_mask.jpeg" if not im else f"img_{id}_out.jpeg")
    
    # masks_np = process_unc_map(masks, False, False, False)
    # masks_np_percentile = [(m > np.percentile(m, 90))*255 for m in masks_np]
    masks_np_percentile = process_unc_map(masks, scale_independent=True, amplify=False, abs=True)
    if im:
        masks_np_percentile = process_unc_map(masks, to_heatmap=False, abs=False, rescale=False)
    
    fig, axs = plt.subplots(1, len(masks_np_percentile), 
                            tight_layout=True, figsize=(60, 20))
    for i, m in enumerate(masks_np_percentile):
        axs[i].imshow(m)
        axs[i].axis('off')
        axs[i].set_title(f'block {i}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
    # masks_np = process_unc_map(masks, False, True)
    # new_out_dir = os.path.join(out_dir, "Mask_Diff")
    # os.makedirs(new_out_dir, exist_ok=True)
    
    # save_file = os.path.join(new_out_dir, f"img_{id}_mask_diff.jpeg")
    # fig, axs = plt.subplots(1, len(masks_np)-1, 
    #                         tight_layout=True, figsize=(60, 20))
    # for i, m in enumerate(masks_np):
    #     if i==len(masks_np)-1: continue
    #     axs[i].imshow((m > masks_np[i+1]).astype(int)*255)
    #     axs[i].axis('off')
    #     axs[i].set_title(f'block {i} - perf {val_perfs[i].detach().item()}')
        
    # plt.savefig(save_file)
    # plt.close(fig)
    # plt.show()
    
def visualize_histogram_im(masks, id):
    
    ims = process_unc_map(masks, to_heatmap=False, rescale=True, abs=True, amplify=False)
    new_out_dir = out_dir
    os.makedirs(new_out_dir, exist_ok=True)
    save_file = os.path.join(new_out_dir, f"img_{id}_hist.jpeg")

    # calculate mean value from RGB channels and flatten to 1D array
    vals = [im.mean(axis=2).flatten() for im in ims]
    # plot histogram with 255 bins
    fig, axs = plt.subplots(1, len(vals), sharey=True, 
                            tight_layout=True, figsize=(60, 20))
    lim = max([max(val) for val in vals])
    for i, val in enumerate(vals):
        axs[i].hist(val, 255, edgecolor='black')
        axs[i].set_xlim(0, 255)
        axs[i].set_title(f'block {i} - mean {val.mean()} - std {np.std(val)}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
def visualize_fusion_map(outs, masks, im_idx, visualize=False, align_biases=None):
    
    save_file = os.path.join(out_dir, f"img_{im_idx}_fusion.jpeg")
    # for m in masks:
    #     print(m.max(), m.min())
    masks = [torch.exp(m) for i, m in enumerate(masks)]
    
    all_masks = torch.stack(masks, dim=-1) # 1xCxHxW -> 1xCxHxWxN
    raw_indices = torch.argmin(all_masks, dim=-1)    # 0->N-1, 1xCxHxW
    onehot_indices = F.one_hot(raw_indices, num_classes=len(masks)).float() # 1xCxHxWxN
    
    filter_outs = process_unc_map(outs, to_heatmap=False, abs=False, rescale=False)
    processed_outs = list()
    percent = np.zeros(shape=[len(filter_outs)])

    if visualize: fig, axs = plt.subplots(ncols=len(filter_outs)+1, nrows=1, figsize=(20, 4))
    for i in range(len(filter_outs) + 1):
        if i<len(filter_outs):
            fout = filter_outs[i]
            p = onehot_indices[..., i].float().mean()
            percent[i] = p
            cur_mask = onehot_indices[..., i].squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
            
            cur_fout = fout*cur_mask
            processed_outs.append(fout * cur_mask)
        else:
            # filter_outs = [f + align_biases[i] * onehot_indices[..., i] if i<len(filter_outs)-1 else f for i, f in enumerate(filter_outs)]
            fout = np.sum(np.stack(processed_outs, axis=0), axis=0)
            cur_mask = np.ones_like(fout).astype(np.uint8)
            p=1
            
        if visualize:
            fout_ = np.clip(fout*cur_mask, 0, 1)
            axs[i].imshow(fout_)
            axs[i].set_title(f"p={p*100:.2f}%|b={i}|")
            
            plt.imsave(os.path.join(out_dir, f"img_{im_idx}_b{i}_fusion.jpeg"), fout_)
            
    if visualize:
        plt.savefig(save_file)
        plt.close(fig)
        plt.show()
    
    fout = torch.tensor(fout).permute(2,0,1).unsqueeze(0).float()
    
    return fout, percent
        
# testing

num_blocks = args.n_resgroups // 2 if args.n_resgroups > 0 else args.n_resblocks // 2 
num_blocks = min(num_blocks, args.n_estimators)

def test():
    core.eval()
    # core.train()
    percent_total = np.zeros(shape=[num_blocks])
    x = data.common.load_image_as_Tensor(args.infer_file)
    x  = x.cuda()

    with torch.no_grad():
        out = core.eunaf_forward(x)
        # out = core(x)
    
    yfs, masks = out
    yf_fuse, percent = visualize_fusion_map(yfs, masks, 0)
    yf_fuse = yf_fuse.cuda()
        
    percent_total += percent
    
    visualize_fusion_map(yfs, masks, 0, visualize=True)
        
    if args.visualize:
        visualize_unc_map(masks, 0)
            # visualize_unc_map_binary(masks, batch_idx, perf_v_layers)
            # visualize_unc_map(yfs, batch_idx, perf_v_layers, True)

if __name__ == '__main__':
    test()