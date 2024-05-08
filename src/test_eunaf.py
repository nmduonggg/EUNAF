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
if args.weight:
    fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}' if args.n_resgroups > 0 \
        else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}'
    out_dir = os.path.join(args.cv_dir, fname)
    args.weight = os.path.join(out_dir, '_best.t7')
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight), strict=True)
core.cuda()

loss_func = loss.create_loss_func(args.loss)

# working dir
# out_dir = os.path.join(args.analyze_dir, args.template, name+f'_nblock{args.nblocks}', args.testset_tag)
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
    
# def visualize_fusion_map(outs, masks, im_idx, align_biases=None):
    
#     save_file = os.path.join(out_dir, f"img_{im_idx}_fusion.jpeg")
#     masks = [torch.mean(torch.exp(m.squeeze(0)), dim=0, keepdim=True) for m in masks]
#     # masks[-1] *= 1.0
#     all_masks = torch.stack(masks, dim=-1) # 1xHxW -> 1xHxWxN
#     raw_indices = torch.argmin(all_masks, dim=-1)    # 0->N-1, 1xHxW
#     onehot_indices = F.one_hot(raw_indices, num_classes=len(masks)).float() # 1xHxWxN
    
#     filter_outs = list()
#     for i, out in enumerate(outs):
#         cur_mask = onehot_indices[..., i]  # 1xHxW binary
#         cur_out = out.squeeze(0) * cur_mask.repeat(3, 1, 1)
#         filter_outs.append(cur_out)   #CxHxW

#     fig, axs = plt.subplots(ncols=len(filter_outs)+1, nrows=1, figsize=(20, 4))
#     for i in range(len(filter_outs) + 1):
#         if i<len(filter_outs):
#             fout = filter_outs[i]
#             p = onehot_indices[..., i].float().mean()
#         else:
#             # filter_outs = [f + align_biases[i] * onehot_indices[..., i] if i<len(filter_outs)-1 else f for i, f in enumerate(filter_outs)]
#             fout = torch.sum(torch.stack(filter_outs, dim=0), dim=0)
#             p=1
            
#         fout = fout.permute(1,2,0)  # CxHxW -> HxWxC
#         fout = fout.detach().cpu().numpy()
#         try:
#             cur_mask = onehot_indices[..., i].permute(1,2,0).cpu().numpy()  # 1xHxW binary
#         except:
#             cur_mask = np.ones_like(fout)
#         fout_ = ((fout*255)*cur_mask).round().astype(np.uint8)
            
#         axs[i].imshow(fout_)
#         axs[i].set_title(f"p={p*100:.2f}% - block={i}")
        
#         plt.imsave(os.path.join(out_dir, f"img_{im_idx}_b{i}_fusion.jpeg"), fout_)
    
#     plt.savefig(save_file)
#     plt.close(fig)
#     plt.show()
    
#     fout = torch.tensor(fout).permute(2,0,1).unsqueeze(0).float()
    
#     return fout

def visualize_fusion_map(outs, masks, im_idx, perfs=[], visualize=False, align_biases=None):
    
    save_file = os.path.join(out_dir, f"img_{im_idx}_fusion.jpeg")
    # for m in masks:
    #     print(m.max(), m.min())
    masks = [torch.exp(m) for i, m in enumerate(masks)]
    # masks[-1] *= 1.08
    # masks[-1] *= torch.mean(masks[0], dim=[2,3], keepdim=True) / torch.mean(masks[-1], dim=[2,3], keepdim=True)
    # new_masks = list()
    # for i in range(len(masks)):
    #     tmp_mask = masks[i]
    #     tmp_mask = (tmp_mask - torch.mean(tmp_mask, dim=[2,3], keepdim=True)) / torch.std(tmp_mask, dim=[2,3], keepdim=True)
    #     new_masks.append(tmp_mask)
    # masks = new_masks
    
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
            
            plt.imsave(os.path.join(out_dir, f"img_{im_idx}_b{i}_fusion.jpeg"), fout_)
    if visualize:
        plt.savefig(save_file)
        plt.close(fig)
        plt.show()
    
    fout = torch.tensor(fout).permute(2,0,1).unsqueeze(0).float()
    
    return fout, percent
        
# testing

t = 5e-3
psnr_unc_map = np.ones((len(XYtest), 12))
num_blocks = args.n_resgroups // 2 if args.n_resgroups > 0 else args.n_resblocks // 2 
num_blocks = min(args.n_estimators, num_blocks)

def test():
    psnrs_val = [0 for _ in range(num_blocks)]
    ssims_val = [0 for _ in range(num_blocks)]
    uncertainty_val = [0 for _ in range(num_blocks)]
    total_val_loss = 0.0
    total_mask_loss = 0.0
    psnr_fuse, ssim_fuse = 0.0, 0.0
    #walk through the test set
    core.eval()
    # core.train()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
    # align_biases = core.align_biases
    percent_total = np.zeros(shape=[num_blocks])
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            out = core.eunaf_forward(x)
            # out = core(x)
        
        yfs, masks = out
        yf_fuse, percent = visualize_fusion_map(yfs, masks, batch_idx)
        yf_fuse = yf_fuse.cuda()
        
        percent_total += percent
        cur_psnr_fuse, cur_ssim_fuse = evaluation.calculate_all(args, yf_fuse, yt)
        psnr_fuse += cur_psnr_fuse
        ssim_fuse += cur_ssim_fuse
        
        if args.visualize:
            visualize_histogram_im(masks, batch_idx)
            # visualize_error_map(yfs, yt, batch_idx)
            # get_error_btw_F(yfs, batch_idx)
            # visualize_unc_enhance(masks, batch_idx)
        
        val_loss = sum([loss_func(yf, yt).item() for yf in yfs]) / num_blocks
            
        perf_v_layers = [evaluation.calculate_all(args, yf, yt) for yf in yfs]
        
        psnr_v_layers, ssim_v_layers = list(), list()
        for i, v in enumerate(perf_v_layers):
            psnr_v_layers.append(v[0])
            ssim_v_layers.append(v[1])
        
        unc_v_layers = [m.mean().cpu().item() for m in masks]
        error_v_layers = [torch.abs(yt-yf).mean().item() for yf in yfs]
        
        visualize_fusion_map(yfs, masks, batch_idx, perfs=psnr_v_layers+[cur_psnr_fuse], visualize=True)
        
        if args.visualize:
            visualize_unc_map(
                [m[:, :1, ...] for m in masks], 
                batch_idx, perf_v_layers)
            
        for i, p in enumerate(psnr_v_layers):
            psnrs_val[i] = psnrs_val[i] + p
            ssims_val[i] += ssim_v_layers[i]
            uncertainty_val[i] = uncertainty_val[i] + torch.exp(masks[i]).contiguous().cpu().mean()
        total_val_loss += val_loss
        
        
    # np_fn = os.path.join(out_dir, f'psn_unc_{args.testset_tag}.npy')
    # np.save(np_fn, psnr_unc_map)

    psnrs_val = [p / len(XYtest) for p in psnrs_val]
    ssims_val = [p / len(XYtest) for p in ssims_val]
    percent_total = percent_total / len(XYtest)
    psnr_fuse = psnr_fuse / len(XYtest)
    ssim_fuse = ssim_fuse / len(XYtest)
    
    print(*psnrs_val, psnr_fuse)
    print(*ssims_val, ssim_fuse)
    
    percent_total = percent_total.tolist()
    for perc in percent_total:
        print( f"{(perc*100):.3f}", end=' ')
    
    uncertainty_val = [u / len(XYtest) for u in uncertainty_val]
    total_val_loss /= len(XYtest)
    total_mask_loss /= len(XYtest)

if __name__ == '__main__':
    test()