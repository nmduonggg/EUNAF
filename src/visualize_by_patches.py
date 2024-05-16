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

if args.template is not None:
    template.set_template(args)

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
# if args.weight:
#     fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}' if args.n_resgroups > 0 \
#         else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}'
#     out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', fname)
#     args.weight = os.path.join(out_dir, '_best.t7')
#     print(f"[INFO] Load weight from {args.weight}")
#     core.load_state_dict(torch.load(args.weight), strict=False)
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
# out_dir = os.path.join(args.analyze_dir, args.template, name+f'_nblock{args.nblocks}', args.testset_tag)
out_dir = os.path.join(args.analyze_dir, "by_patches", name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}', args.testset_tag)
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

def visualize_unc_map(masks, id, val_perfs, im=False):
    new_out_dir = out_dir
    os.makedirs(new_out_dir, exist_ok=True)
    save_file = os.path.join(new_out_dir, f"img_{id}_mask.jpeg" if not im else f"img_{id}_out.jpeg")
    
    # masks_np = process_unc_map(masks, False, False, False)
    # masks_np_percentile = [(m > np.percentile(m, 90))*255 for m in masks_np]
    masks_np_percentile = process_unc_map(masks, scale_independent=True, amplify=True, abs=True)
    if im:
        masks_np_percentile = process_unc_map(masks, to_heatmap=False, abs=False, rescale=False)
    
    fig, axs = plt.subplots(1, len(masks_np_percentile), 
                            tight_layout=True, figsize=(60, 20))
    for i, m in enumerate(masks_np_percentile):
        axs[i].imshow(m)
        axs[i].axis('off')
        axs[i].set_title(f'block {i} - perf {val_perfs[i].detach().item()}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    

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
        axs[i].hist(val, 100, edgecolor='black')
        axs[i].set_xlim(0, 100)
        axs[i].set_title(f'block {i} - mean {val.mean()} - std {np.std(val)}')
        
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
def get_error_btw_F(yfs, id):
    error_track = []
    all_errors = []
    
    for i in range(len(yfs)):
        if i==len(yfs)-1: continue
        error = torch.abs(yfs[i+1]-yfs[i])
        all_errors.append(error)
    all_errors = torch.stack(all_errors)
    pmin = torch.min(all_errors)
    pmax = torch.max(all_errors)
    
    fig, axs = plt.subplots(1, len(yfs)-1, figsize=(40, 10))
    for i in range(len(yfs)):
        if i==len(yfs)-1: continue
        error = torch.abs(yfs[i+1] - yfs[i])
        error = error.squeeze(0).permute(1,2,0)
        
        print(f"Difference in image {id} - block {i} to {i+1} is {error.mean():.9f}")
        error_track.append(error.mean())
        
        error = (error - pmin) / (pmax - pmin)
        error = error.cpu().numpy()
        error = (error*255).round().astype(np.uint8)
        
        new_out_dir = os.path.join(out_dir, "mask_to_NEXT")
        if not os.path.exists(new_out_dir):
            os.makedirs(new_out_dir)
        axs[i].imshow(error)
        axs[i].set_title(f"{i}_to_{i+1}")
        
    save_file = os.path.join(new_out_dir, f"img_{id}_error_btw.jpeg")
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()

    return error_track

def visualize_error_enhance(error_wgt, id):
    enhance_map = []
    new_out_dir = os.path.join(out_dir, "Error_Enhanced")
    os.makedirs(new_out_dir, exist_ok=True)
    for i, e in enumerate(error_wgt):
        if i==len(error_wgt)-1: continue
        e = e.cpu().numpy()
        e1 = error_wgt[i+1].cpu().numpy()
        enhance_e = e - e1
        enhance_map.append((enhance_e > np.percentile(enhance_e, 90)).astype(np.uint8)*255)
    
    fig, axs = plt.subplots(1, len(enhance_map), sharey=True, 
                            tight_layout=True, figsize=(60, 20))
    for i, e in enumerate(enhance_map):
        axs[i].imshow(e)
        axs[i].set_title(f'{i}_to{i+1}')
        
    save_file = os.path.join(new_out_dir, f"img_{id}_enhance_error.jpeg")
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
def visualize_error_map(yfs, yt, id):
    errors = []
    save_file = os.path.join(out_dir, f"img_{id}_error.jpeg")
    for yf in yfs:
        error = torch.abs(yt - yf)
        error = error.squeeze(0)
        error = error.permute(1,2,0)
        errors.append(error)
        
    # visualize_error_enhance(errors, id)
        
    pmin = torch.min(torch.stack(errors))
    pmax = torch.max(torch.stack(errors))
    
    ep = []
    for e in errors:
        e = (e - pmin) / (pmax - pmin)
        e = e.cpu().numpy()
        e = (e * 255.).astype(np.uint8)
        ep.append(e)
        
    fig, axs = plt.subplots(1, len(ep), sharey=True, 
                            tight_layout=True, figsize=(60, 20))
    for i, e in enumerate(ep):
        # e = (e > np.percentile(e, 75)).astype(np.uint8) * 255
        e = cv2.applyColorMap(e, cv2.COLORMAP_JET)
        e = cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
        axs[i].imshow(e)
        axs[i].set_title(f'block {i} - error {e.mean()}')
    
    plt.savefig(save_file)
    plt.close(fig)
    plt.show()
    
def get_fusion_map_last(outs, masks, rates=[]):
    
    masks = [torch.mean(torch.exp(m), dim=1, keepdim=True) for i, m in enumerate(masks)]
    mask = masks[-1].clone().detach().cpu()    # B, C, H, W
    bs, _, h, w = mask.shape
    
    quantile_class = list()
    for r in rates:
        if r > 1: r /= 100
        tmp_mask = mask.squeeze(1).reshape(bs, -1)  # bx(hw)
        q = torch.quantile(tmp_mask, r, dim=1, keepdim=True)
        q = q.reshape(bs, 1, 1, 1)
        quantile_class.append(q)
    
    per_class = list()
    for i in range(len(quantile_class)+1):
        q = quantile_class[i] if i<len(quantile_class) else quantile_class[i-1]
        q = torch.ones_like(mask) * q
        if i==0:
            p = (mask < q).float()
        elif i==len(quantile_class):
            p = (q <= mask).float()
        else:
            p = (torch.logical_and(quantile_class[i-1] <= mask, mask < q)).float()
        per_class.append(p)
        
    processed_outs = list()

    for i in range(len(outs) + 1):
        if i<len(outs):
            fout = outs[i]
            cur_mask = per_class[i].to(fout.device)
            cur_fout = fout*cur_mask
            processed_outs.append(fout * cur_mask)
            
        else:
            # filter_outs = [f + align_biases[i] * onehot_indices[..., i] if i<len(filter_outs)-1 else f for i, f in enumerate(filter_outs)]
            fout = torch.sum(torch.stack(processed_outs, dim=0), dim=0)
    
    fout = fout.float()
    
    return fout
    
def visualize_fusion_map(outs, masks, im_idx, perfs=[], visualize=False, align_biases=None):
    
    save_file = os.path.join(out_dir, f"img_{im_idx}_fusion.jpeg")
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
            axs[i].set_title(f"p={p*100:.2f}%|b={i}|acc={round(perfs[i].item(), 4)}")
            
            # plt.imsave(os.path.join(out_dir, f"img_{im_idx}_b{i}_fusion.jpeg"), fout_)
    if visualize:
        plt.savefig(save_file)
        plt.close(fig)
        plt.show()
    
    fout = torch.tensor(fout).permute(2,0,1).unsqueeze(0).float()
    
    return fout, percent

def visualize_fusion_map_by_errors(outs, yt, im_idx):
    
    errors = [torch.abs(out-outs[-1]) for out in outs]
    
    all_masks = torch.stack(errors, dim=-1) # 1xCxHxW -> 1xCxHxWxN
    raw_indices = torch.argmin(all_masks, dim=-1)    # 0->N-1, 1xCxHxW
    onehot_indices = F.one_hot(raw_indices, num_classes=len(errors)).float() # 1xCxHxWxN
    
    filter_outs = process_unc_map(outs, to_heatmap=False, abs=False, rescale=False)
    processed_outs = list()
    percent = np.zeros(shape=[len(filter_outs)])

    for i in range(len(filter_outs) + 1):
        if i<len(filter_outs):
            fout = filter_outs[i]
            p = onehot_indices[..., i].float().mean()
            percent[i] = p
            cur_mask = onehot_indices[..., i].squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8)
            
            cur_fout = fout*cur_mask
            processed_outs.append(fout * cur_mask)
        else:
            fout = np.sum(np.stack(processed_outs, axis=0), axis=0)
            cur_mask = np.ones_like(fout).astype(np.uint8)
            p=1
            fout_ = np.clip(fout*cur_mask, 0, 1)
            
            # plt.imsave(os.path.join(out_dir, f"img_{im_idx}_fusion_by_error.jpeg"), fout_)
    
    fout = torch.tensor(fout).permute(2,0,1).unsqueeze(0).float()
    
    return fout, percent

def visualize_classified_patch_level(p_yfs, p_masks, im_idx):
    # p_yfs, p_masks: all patches of yfs and masks of all num block stages [[HxWxC]*n_patches]xnum_blocks
    yfs = [np.stack(pm, axis=0) for pm in p_yfs]  # PxHxWxC
    masks = [
        np.stack([
            np.mean(pm) for pm in bm], axis=0) for bm in p_masks] 
    # masks[-1] *= 1.1
    
    all_masks = torch.tensor(np.stack(masks, axis=-1)) # P -> PxN
    raw_indices = torch.argmin(all_masks, dim=-1)    # 0->N-1, P
    onehot_indices = F.one_hot(raw_indices, num_classes=len(masks)).float() # PxN
    
    processed_outs = 0
    class_colors = [
        [40, 66, 235],
        [19, 239, 85],
        [235, 255, 128],
        [255, 0, 0]
    ]
    for i in range(len(p_masks) ):
        fout = np.ones_like(yfs[i]) * np.array(class_colors[i])/255.0
        fout[:, :1, :, :] = 0
        fout[:, -1:, :, :] = 0
        fout[:, :, :1, :] = 0
        fout[:, :, -1:, :] = 0
        cur_mask = onehot_indices[..., i].numpy().astype(np.uint8)
        cur_mask = cur_mask.reshape(-1, 1, 1, 1)
        
        cur_fout = (fout*cur_mask)
        processed_outs += cur_fout
    
    classified_map = [processed_outs[i,...] for i in range(processed_outs.shape[0])]
    
    return classified_map

def visualize_edge_map(patches, im_idx, scale):
    
    # masks = [torch.mean(torch.exp(m), dim=1, keepdim=True) for m in masks]
    # masks = [torch.log(m) for m in masks]
    patches = [(p*255).astype(np.uint8) for p in patches]   # PxHxWxC
    patches = [cv2.resize(img, [img.shape[1] * scale, img.shape[0] * scale], interpolation=cv2.INTER_CUBIC)
            for img in patches]
    patches_np = np.array(patches)
    imscores = np.array([utils.laplacian(p).mean() for p in patches])   # P
    
    q1, q2, q3 = np.percentile(imscores, [20, 30, 50])
    
    p0 = (imscores < q1).astype(int)
    p1 = (np.logical_and(q1 <= imscores, imscores < q2)).astype(int)
    p2 = (np.logical_and(q2 <= imscores, imscores < q3)).astype(int)
    p3 = (q3 <= imscores).astype(int)
    class_colors = [
        [0, 0, 255],
        [19, 239, 85],
        [235, 255, 128],
        [255, 0, 0]
    ]

    out_patches = 0
    per_class = [p0, p1, p2, p3]
    for i, class_mask in enumerate(per_class):
        color = class_colors[i]
        color_np = np.ones_like(patches_np) * np.array(color).reshape(1, 1, 1, -1)/255
        color_np[:, :1, :, :] = 0
        color_np[:, -1:, :, :] = 0
        color_np[:, :, :1, :] = 0
        color_np[:, :, -1:, :] = 0
        out_patches += color_np * class_mask.reshape(-1, 1, 1, 1)
    
    out_patches = [out_patches[i, :, :, :] for i in range(len(patches))]
    return out_patches
def visualize_last_unc_map(patches, im_idx, last_unc):
    
    # masks = [torch.mean(torch.exp(m), dim=1, keepdim=True) for m in masks]
    # masks = [torch.log(m) for m in masks]
    patches = [(p*255).astype(np.uint8) for p in patches]   # PxHxWxC
    patches_np = np.array(patches)
    imscores = np.array([u.mean() for u in last_unc])
    
    q1, q2, q3 = np.percentile(imscores, [20, 30, 50])
    
    p0 = (imscores < q1).astype(int)
    p1 = (np.logical_and(q1 <= imscores, imscores < q2)).astype(int)
    p2 = (np.logical_and(q2 <= imscores, imscores < q3)).astype(int)
    p3 = (q3 <= imscores).astype(int)
    class_colors = [
        [0, 0, 255],
        [19, 239, 85],
        [235, 255, 128],
        [255, 0, 0]
    ]
    per_class = [p0, p1, p2, p3]
    out_patches = 0
    for i, class_mask in enumerate(per_class):
        color = class_colors[i]
        color_np = np.ones_like(patches_np) * np.array(color).reshape(1, 1, 1, -1)/255
        color_np[:, :1, :, :] = 0
        color_np[:, -1:, :, :] = 0
        color_np[:, :, :1, :] = 0
        color_np[:, :, -1:, :] = 0
        out_patches += color_np * class_mask.reshape(-1, 1, 1, 1)
    
    out_patches = [out_patches[i, :, :, :] for i in range(len(patches))]
    return out_patches, per_class

def fuse_by_last_unc_by_patches(all_levels, im_idx, patch_indices):
    all_level_patches = list()
    for patches in all_levels:
        patches_np = np.array(patches)
        all_level_patches.append(patches_np)
        
    fusion_final = 0
    for i, pid in enumerate(patch_indices):
        fuse_pid = pid.reshape(-1, 1, 1, 1) * all_level_patches[i]
        fusion_final += fuse_pid
    fusion_final = [fusion_final[i,:,:,:] for i in range(len(patches))]
    return fusion_final
        

def visualize_last_psnr_map(patches, im_idx, psnrs):
    
    # masks = [torch.mean(torch.exp(m), dim=1, keepdim=True) for m in masks]
    # masks = [torch.log(m) for m in masks]
    patches = [(p*255).astype(np.uint8) for p in patches]   # PxHxWxC
    patches_np = np.array(patches)
    imscores = np.array([u.mean() for u in psnrs])
    # [40, 60, 80]
    q1, q2, q3 = np.percentile(imscores, [50, 70, 90])
    
    p0 = (imscores <= q1).astype(int)
    p1 = (np.logical_and(q1 < imscores, imscores <= q2)).astype(int)
    p2 = (np.logical_and(q2 < imscores, imscores <= q3)).astype(int)
    p3 = (q3 < imscores).astype(int)
    
    per_class = [p0, p1, p2, p3]
    class_colors = [
        [255, 0, 0],
        [235, 255, 128],
        [19, 239, 85],
        [0, 0, 255]
    ]

    out_patches = 0
    for i, class_mask in enumerate(per_class):
        color = class_colors[i]
        color_np = np.ones_like(patches_np) * np.array(color).reshape(1, 1, 1, -1)/255
        color_np[:, :1, :, :] = 0
        color_np[:, -1:, :, :] = 0
        color_np[:, :, :1, :] = 0
        color_np[:, :, -1:, :] = 0
        out_patches += color_np * class_mask.reshape(-1, 1, 1, 1)
    
    out_patches = [out_patches[i, :, :, :] for i in range(len(patches))]
    return out_patches
        
# testing

t = 5e-3
psnr_unc_map = np.ones((len(XYtest), 12))
num_blocks = args.n_resgroups // 2 if args.n_resgroups > 0 else args.n_resblocks // 2 
num_blocks = min(args.n_estimators, num_blocks)

patch_size = 32
step = 32
alpha = 0.7

def test():
    psnrs_val = [0 for _ in range(num_blocks)]
    ssims_val = [0 for _ in range(num_blocks)]
    uncertainty_val = [0 for _ in range(num_blocks)]
    total_val_loss = 0.0
    total_mask_loss = 0.0
    psnr_fuse, ssim_fuse = 0.0, 0.0
    psnr_fuse_err, ssim_fuse_err = 0.0, 0.0
    psnr_fuse_unc, ssim_fuse_unc = 0.0, 0.0
    
    #walk through the test set
    core.eval()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
            
    # flops of 1 patch
    utils.calc_flops(core, (1, 3, 32, 32))
            
    percent_total = np.zeros(shape=[num_blocks])
    percent_total_err = np.zeros(shape=[num_blocks])
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        # x  = x.cuda()
        # yt = yt.cuda()
        
        # yt = utils.resize_image_tensor(x, yt, args.scale, args.rgb_range)
        # yt = utils.modcrop(yt)
        
        # cut patches
        x_np = x.permute(0,2,3,1).squeeze(0).numpy()
        lr_list, num_h, num_w, h, w = utils.crop_cpu(x_np, patch_size, step)
        yt = yt[:, :, :h*args.scale, :w*args.scale]
        y_np = yt.permute(0,2,3,1).squeeze(0).numpy()
        hr_list = utils.crop_cpu(y_np, patch_size * args.scale, step*args.scale)[0]
        
        
        
        
        combine_img_lists = [list() for _ in range(num_blocks)]
        combine_unc_lists = [list() for _ in range(num_blocks)]
        all_last_psnrs = list()
        for lr_img, hr_img in zip(lr_list, hr_list):
            img = lr_img.astype(np.float32) 
            img = img[:, :, :3]
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).cuda()
            
            gt = hr_img.astype(np.float32) 
            gt = gt[:, :, :3]
            gt = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).cuda()
        
            with torch.no_grad():
                out = core.eunaf_forward(img)
            
            p_yfs, p_masks = out
            cur_psnr = evaluation.calculate(args, p_yfs[-1], gt)
            all_last_psnrs.append(cur_psnr)
            
            for i in range(len(p_yfs)):
                combine_img_lists[i].append(p_yfs[i].cpu().squeeze(0).permute(1,2,0).numpy())
                combine_unc_lists[i].append(p_masks[i].cpu().squeeze(0).permute(1,2,0).numpy())
            
        yfs, masks = list(), list()
        for i in range(num_blocks):
            yfs.append(
                torch.from_numpy(utils.combine(combine_img_lists[i], num_h, num_w, h, w, patch_size, step, args.scale)).permute(2,0,1).unsqueeze(0))
            masks.append(
                torch.from_numpy(utils.combine(combine_unc_lists[i], num_h, num_w, h, w, patch_size, step, args.scale)).permute(2,0,1).unsqueeze(0))
            
        yf_fuse, percent = visualize_fusion_map(yfs, masks, batch_idx)
        # yf_fuse_by_err, percent_err = visualize_fusion_map_by_errors(yfs, yt, batch_idx)
        
        percent_total += percent
        # percent_total_err += percent_err
        cur_psnr_fuse, cur_ssim_fuse = evaluation.calculate_all(args, yf_fuse, yt)
        # cur_psnr_fuse_err, cur_ssim_fuse_err = evaluation.calculate_all(args, yf_fuse_by_err, yt)
        psnr_fuse += cur_psnr_fuse
        ssim_fuse += cur_ssim_fuse
        # psnr_fuse_err += cur_psnr_fuse_err
        # ssim_fuse_err += cur_ssim_fuse_err
        
        val_loss = sum([loss_func(yf, yt).item() for yf in yfs]) / num_blocks
            
        perf_v_layers = [evaluation.calculate_all(args, yf, yt) for yf in yfs]
        
        psnr_v_layers, ssim_v_layers = list(), list()
        for i, v in enumerate(perf_v_layers):
            psnr_v_layers.append(v[0])
            ssim_v_layers.append(v[1])
        
        unc_v_layers = [m.mean().cpu().item() for m in masks]
        # error_v_layers = [torch.abs(yt-yf).mean().item() for yf in yfs]
        
        # if args.visualize:
        #     visualize_fusion_map(yfs, masks, batch_idx, perfs=psnr_v_layers+[cur_psnr_fuse], visualize=True)
        
        unc_patch_map, unc_patch_indices = visualize_last_unc_map(combine_img_lists[-1], batch_idx, combine_unc_lists[-1])
        fused_yf_unc_patches = fuse_by_last_unc_by_patches(combine_img_lists, batch_idx, unc_patch_indices)
        fused_yf_tensor = torch.from_numpy(utils.combine(fused_yf_unc_patches, num_h, num_w, h, w, patch_size, step, args.scale)).permute(2,0,1).unsqueeze(0)
        cur_unc_psnr, cur_unc_ssim = evaluation.calculate_all(args, fused_yf_tensor, yt)
        psnr_fuse_unc += cur_unc_psnr
        ssim_fuse_unc += cur_unc_ssim
        
        if args.visualize:
            # classified_patch_map = visualize_classified_patch_level(combine_img_lists, combine_unc_lists, batch_idx)
            # classified_map = utils.combine(classified_patch_map, num_h, num_w, h, w, patch_size, step, args.scale)
            # # plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_classify_patch.jpeg"), classified_map)
            # yt_image = yt.squeeze(0).cpu().permute(1,2,0).numpy()
            # masked_yt = utils.apply_alpha_mask(yt_image, classified_map, alpha)
            # plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_alpha_masked.jpeg"), masked_yt)
            
            edge_patch_map = visualize_edge_map(lr_list, batch_idx, args.scale)
            edge_map = utils.combine(edge_patch_map, num_h, num_w, h, w, patch_size, step, args.scale)
            # plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_edge_patch.jpeg"), edge_map)
            yt_image = yt.squeeze(0).cpu().permute(1,2,0).numpy()
            masked_yt_edge = utils.apply_alpha_mask(yt_image, edge_map, alpha)
            plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_edge_alpha_masked.jpeg"), masked_yt_edge)
            
            unc_patch_map, unc_patch_indices = visualize_last_unc_map(combine_img_lists[-1], batch_idx, combine_unc_lists[-1])
            unc_map = utils.combine(unc_patch_map, num_h, num_w, h, w, patch_size, step, args.scale)
            # plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_unc_patch.jpeg"), unc_map)
            yt_image = yt.squeeze(0).cpu().permute(1,2,0).numpy()
            masked_yt_unc = utils.apply_alpha_mask(yt_image, unc_map, alpha)
            plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_unc_alpha_masked.jpeg"), masked_yt_unc)

            psnr_patch_map = visualize_last_psnr_map(combine_img_lists[-1], batch_idx, all_last_psnrs)
            psnr_map = utils.combine(psnr_patch_map, num_h, num_w, h, w, patch_size, step, args.scale)
            # plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_psnr_patch.jpeg"), psnr_map)
            yt_image = yt.squeeze(0).cpu().permute(1,2,0).numpy()
            masked_yt_psnr = utils.apply_alpha_mask(yt_image, psnr_map, alpha)
            plt.imsave(os.path.join(out_dir, f"img_{batch_idx}_psnr_alpha_masked.jpeg"), masked_yt_psnr)
            
        for i, p in enumerate(psnr_v_layers):
            psnrs_val[i] = psnrs_val[i] + p
            ssims_val[i] += ssim_v_layers[i]
            uncertainty_val[i] = uncertainty_val[i] + torch.exp(masks[i]).contiguous().cpu().mean()
        total_val_loss += val_loss

    psnrs_val = [p / len(XYtest) for p in psnrs_val]
    ssims_val = [p / len(XYtest) for p in ssims_val]
    percent_total = percent_total / len(XYtest)
    # percent_total_err /= len(XYtest)
    psnr_fuse = psnr_fuse / len(XYtest)
    ssim_fuse = ssim_fuse / len(XYtest)
    
    # psnr_fuse_err /= len(XYtest) 
    # ssim_fuse_err /= len(XYtest)
    
    psnr_fuse_unc /= len(XYtest)
    ssim_fuse_unc /= len(XYtest)
    
    print(*psnrs_val, psnr_fuse)
    print(*ssims_val, ssim_fuse)
    
    # print("error psnr: ", psnr_fuse_err)
    # print("error ssim: ", ssim_fuse_err)
    
    print("unc psnr: ", psnr_fuse_unc)
    print("unc ssim: ", ssim_fuse_unc)
    
    percent_total = percent_total.tolist()
    # percent_total_err = percent_total_err.tolist()
    for perc in percent_total:
        print( f"{(perc*100):.3f}", end=' ')
    print()
    # for perc in percent_total_err:
    #     print( f"{(perc*100):.3f}", end=' ')
    
    uncertainty_val = [u / len(XYtest) for u in uncertainty_val]
    total_val_loss /= len(XYtest)
    total_mask_loss /= len(XYtest)

if __name__ == '__main__':
    test()