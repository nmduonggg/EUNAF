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
    out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', 'Error-predict', '1est', fname)
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
    
num_blocks = args.n_resgroups // 2 if args.n_resgroups > 0 else args.n_resblocks // 2 
num_blocks = min(args.n_estimators, num_blocks)
num_blocks = 3

psnr_map = np.zeros((len(XYtest), num_blocks))
ssim_map = np.zeros((len(XYtest), num_blocks))
unc_map = np.zeros((len(XYtest), num_blocks))
edge_map = np.zeros((len(XYtest)))

def laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def test(r):
    psnrs_val = [0 for _ in range(num_blocks)]
    ssims_val = [0 for _ in range(num_blocks)]
    uncertainty_val = [0 for _ in range(num_blocks)]
    
    core.eval()
    # core.train()
    torch.manual_seed(0)
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        
        x = x + torch.randn_like(x) * r
        
        x_np = (x.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        edge = laplacian(x_np).mean()
        edge_map[batch_idx] = edge
        
        x  = x.cuda()
        yt = yt.cuda()
        with torch.no_grad():
            out = core.eunaf_forward(x)
        
        yfs, masks = out
        perf_v_layers = [evaluation.calculate_all(args, yf, yt) for yf in yfs]
        
        psnr_v_layers, ssim_v_layers = list(), list()
        for i, v in enumerate(perf_v_layers):
            psnr_v_layers.append(v[0].item())
            ssim_v_layers.append(v[1].item())
        unc_v_layers = [masks[:, i].cpu().item() for i in range(masks.shape[1])]
        
        # store value
        psnr_map[batch_idx, :] = np.array(psnr_v_layers).reshape(1, -1)
        ssim_map[batch_idx, :] = np.array(ssim_v_layers).reshape(1, -1)
        unc_map[batch_idx, :] = np.array(unc_v_layers).reshape(1, -1)
            
        for i, p in enumerate(psnr_v_layers):
            psnrs_val[i] = psnrs_val[i] + p
            ssims_val[i] += ssim_v_layers[i]
        
    # save each file
    np.save(os.path.join(out_dir, f'psnr_{args.testset_tag}_{r}.npy'), psnr_map)
    np.save(os.path.join(out_dir, f'ssim_{args.testset_tag}_{r}.npy'), ssim_map)
    np.save(os.path.join(out_dir, f'unc_{args.testset_tag}_{r}.npy'), unc_map)
    np.save(os.path.join(out_dir, f'edge_{args.testset_tag}_{r}.npy'), edge_map)
    

    psnrs_val = [p / len(XYtest) for p in psnrs_val]
    ssims_val = [p / len(XYtest) for p in ssims_val]
    
    print(*psnrs_val)
    print(*ssims_val)
    
    uncertainty_val = [u / len(XYtest) for u in uncertainty_val]

if __name__ == '__main__':
    noise_rates = [0.0, 0.01, 0.02]
    for r in noise_rates:
        print("Gaussian noise variance: ", r)
        test(r)