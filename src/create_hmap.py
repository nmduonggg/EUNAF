import os
import torch
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math

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

cost_dict = {
    'srresnet': [0, 2.04698, 3.66264, 5.194], 
}
baseline_cost_dict = {
    'srresnet': 5.194
}

cost_ees = cost_dict[args.backbone_name]
baseline_cost = baseline_cost_dict[args.backbone_name]

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    fname = name + f"_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}" if args.n_resgroups > 0 \
        else name + f"_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}"
    out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', 'Error-predict', '1est', fname)
    args.weight = os.path.join(out_dir, '_best.t7')
    print(f"[INFO] Load weight from {args.weight}")
    core.load_state_dict(torch.load(args.weight), strict=True)
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

# testing
num_blocks = 4
hmap_model = {
    'hmap_list': [], 'hmap_x_list': [],
    'hmap_y_list': [], 'hmap_y_mean_list': []
}

def test():

    perfs_val = [[] for _ in range(num_blocks)]
    patch_imscores = []
    #walk through the test set
    core.eval()
    for m in core.modules():
        if hasattr(m, '_prepare'):
            m._prepare()
    for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
        # if batch_idx==10: break
        img = x.clone().squeeze(0).permute(1,2,0).numpy() * 255
        imscore = utils.laplacian(img.astype(np.uint8)).mean()
        patch_imscores.append(imscore)
        
        x  = x.cuda()
        yt = yt.cuda()

        with torch.no_grad():
            out = core.eunaf_forward(x)
        
        outs_mean, masks = out
        perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
        
        rm = False
        for i in range(len(perf_layers_mean)):
            curr_perf = perf_layers_mean[i]
            mask0 = torch.where(curr_perf != float("inf") and curr_perf == curr_perf,
                                1., 0.).bool()
            perf_layers_mean[i] = perf_layers_mean[i][mask0]
            rm = not mask0.item() or rm
        
        if rm:  
            patch_imscores = patch_imscores[:-1]    # remove last add-in
        
        if not rm:
            for i, p in enumerate(perf_layers_mean):
                perfs_val[i].append(p.item())
            
    # create hmap
    for ee_id, patch_psnrs in enumerate(perfs_val):
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        
        print(all(np.isfinite(np.array(patch_imscores))), all(np.isfinite(np.array(patch_psnrs))))
        
        hmap, hmap_x, hmap_y, _ = ax.hist2d(patch_imscores, patch_psnrs, bins=30)
        ret = stats.binned_statistic(patch_imscores, patch_psnrs, 'mean', bins=hmap_x)
        hmap_y_mean = ret.statistic   
        
        hmap_x = hmap_x[:-1] + (hmap_x[1]-hmap_x[0])/2
        hmap_y = hmap_y[:-1] + (hmap_y[1]-hmap_y[0])/2
        
        utils.save_hmap(hmap, hmap_x, hmap_y, os.path.join('experiments', f'hmap_{ee_id}.png'), bins=30)
        
        plt.cla()
        plt.close("all")
        
        hmap_model['hmap_list'].append(hmap)
        hmap_model['hmap_x_list'].append(hmap_x)
        hmap_model['hmap_y_list'].append(hmap_y)
        hmap_model['hmap_y_mean_list'].append(hmap_y_mean)
    
    import pickle
    fn = os.path.join('experiments', f'hmap_model.pkl')
    with open(fn, 'wb') as handle:
        pickle.dump(hmap_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    test()