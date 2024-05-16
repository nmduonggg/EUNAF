"""
Train EUNAF-version of backbone SISR network
"""

import os
import torch
import numpy as np
import torch.utils.data as torchdata
import torch.nn.functional as F
import tqdm
import wandb

#custom modules
import data
import evaluation
import loss
import model as supernet
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import utils
from option import parser
from template import train_template as template


args = parser.parse_args()

if args.template is not None:
    template.set_template(args)

print('[INFO] load trainset "%s" from %s' % (args.trainset_tag, args.trainset_dir))
trainset = data.load_trainset(args)
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0)

# model
arch = args.core.split("-")
name = args.template
core = supernet.config(args)
if args.weight:
    fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage-1}' if args.n_resgroups > 0 \
        else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage-1}'
    out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', fname)
    if os.path.exists(out_dir):
        args.weight = os.path.join(out_dir, '_best.t7')
        print(f"[INFO] Load weight from {args.weight}")
        core.load_state_dict(torch.load(args.weight), strict=False)
    args.weight = './checkpoints/PRETRAINED/FSRCNN/FSRCNN_branch3.pth'
    core.load_state_dict(torch.load(args.weight), strict=False)
    print(f"[INFO] Load weight from {args.weight}")
    
core.cuda()

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch
num_blocks = args.n_resgroups if args.n_resgroups > 0 else args.n_resblocks
num_blocks = min(num_blocks//2, args.n_estimators)

optimizer = Adam(core.parameters(), lr=lr, weight_decay=args.weight_decay)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
early_stopper = utils.EarlyStopper(patience=15)
loss_func = loss.create_loss_func(args.loss)

# working dir
fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}' if args.n_resgroups > 0 \
    else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}'
out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', fname)
os.makedirs(out_dir, exist_ok=True)
print("Load ckpoint to: ", out_dir)
    
def get_error_btw_F(yfs):
    error_track = []
    for i in range(len(yfs)):
        if i>=len(yfs)-1: continue
        high_y = yfs[i+1].contiguous()
        low_y = yfs[i].contiguous()
        error_map = torch.abs(high_y - low_y)
        
        max_ = torch.amax(error_map, dim=1, keepdim=True).to(error_map.device)
        min_ = torch.amin(error_map, dim=1, keepdim=True).to(error_map.device)
        eta = (torch.ones_like(max_-min_)*1e-6).to(error_map.device)
        
        error_map = ((error_map - min_ + eta) / (max_ - min_ + eta)).type(torch.FloatTensor)
        error_track.append(error_map)
    
    return error_track

def loss_esu(yfs, masks, yt, freeze_mask=False):
    assert len(yfs)==len(masks), "yfs contains {%d}, while masks contains {%d}" % (len(yfs), len(masks))
    esu = 0.0
    # mean_mask = torch.mean(torch.cat(masks, dim=0), dim=0)
    # yt = yt.repeat(mean_yf.shape[0], 1, 1, 1)
    ori_yt = yt.clone()
    all_masks = torch.exp(torch.stack(masks, dim=0)).clone().detach()
    pmin = torch.amin(all_masks, dim=0)
    pmax = torch.amax(all_masks, dim=0)
    
    for i in range(len(yfs)):
        yf = yfs[i]
        # esu += loss_func(yf, yt) 
        if freeze_mask:
            mask_ = masks[i].clone().detach()
            mask_ = (mask_ - pmin) / (pmax - pmin)  # 0-1 scaling
        else:
            mask_ = masks[i]
        
        s = torch.exp(-mask_)
        yf = torch.mul(yf, s)
        yt = torch.mul(ori_yt, s)
        sl1_loss = loss_func(yf, yt)
        esu = esu + (2*mask_.mean() + sl1_loss)
        
    return esu

def loss_alignment(yfs, masks, yt, align_biases, trainable_mask=False):
    
    final_mask = masks[-1].clone().detach() 
    pmin = torch.amin(final_mask, dim=[2,3], keepdim=True)
    pmax = torch.amax(final_mask, dim=[2,3], keepdim=True)
    final_mask = (final_mask - pmin) / (pmax - pmin+ 1e-9)
    aln_loss_2 = 0.0
    for i, yf in enumerate(yfs):
        if i==len(yfs)-1: continue
        yf_ = yf * final_mask
        yt_ = yt * final_mask 
        aln_loss_2 += loss_func(yf_, yt_)
    aln_loss_2 / (len(yfs)-1)
    
    all_masks = torch.stack(masks, dim=-1) # BxCxHxWxN
    all_masks = all_masks.clone().detach()
    raw_indices = torch.argmin(all_masks, dim=-1)    # BxCxHxW
    onehot_indices = F.one_hot(raw_indices, num_classes=len(masks)).float() # Bx1xHxWx4
        
    fused_out = torch.zeros_like(yfs[0])
    for i, yf in enumerate(yfs):
        onehot = onehot_indices[..., i] # Bx1xHxWxN -> Bx1xHxW
        yf = yf * onehot
        fused_out = fused_out + yf
    
    # aln_loss = loss_func(fused_out, yt) + aln_loss_2*0.5
    aln_loss = loss_func(fused_out, yt)
    
    return aln_loss, fused_out

def rescale_masks(masks):
    new_masks = []
    for m in masks:
        pmin = torch.amin(m, dim=1, keepdim=True)
        m = m - pmin + 1
        m.requires_grad = False
        new_masks.append(m)
    
    return new_masks

def get_fusion_map_last(outs, masks, rates=[]):
    
    masks = [torch.mean(m, dim=1, keepdim=True) for i, m in enumerate(masks)]
    mask = masks[-1].clone().detach().cpu()    # B, C, H, W
    bs, _, h, w = mask.shape
    small_mask = F.interpolate(mask, scale_factor=0.25, mode='bilinear')
    
    quantile_class = list()
    for r in rates:
        if r > 1: r /= 100
        tmp_mask = small_mask.squeeze(1).reshape(bs, -1)  # bx(hw)
        q = torch.quantile(tmp_mask, r, dim=1, keepdim=True)
        q = q.reshape(bs, 1, 1, 1)
        
        quantile_class.append(q)
    
    per_class = list()
    for i in range(len(quantile_class)+1):
        q = quantile_class[i] if i<len(quantile_class) else quantile_class[i-1]
        q = torch.ones_like(small_mask) * q
        if i==0:
            p = (small_mask < q).float()
        elif i==len(quantile_class):
            p = (q <= small_mask).float()
        else:
            p = (torch.logical_and(quantile_class[i-1] <= small_mask, small_mask < q)).float()
        per_class.append(p)
        
    processed_outs = list()

    for i in range(len(outs) + 1):
        if i<len(outs):
            fout = outs[i]
            cur_mask = per_class[i].to(fout.device)
            cur_fout = fout * F.interpolate(cur_mask, scale_factor=4, mode='nearest')
            processed_outs.append(cur_fout)
            
        else:
            # filter_outs = [f + align_biases[i] * onehot_indices[..., i] if i<len(filter_outs)-1 else f for i, f in enumerate(filter_outs)]
            fout = torch.sum(torch.stack(processed_outs, dim=0), dim=0)
    
    fout = fout.float()
    
    return fout

def loss_alignment_2(yfs, masks, yt):
    
    all_rates = [
        [20, 40, 60]
    ]
    aln_loss = 0.0
    for rate in all_rates:
        fused_out = get_fusion_map_last(yfs, masks, rates=rate)
        aln_loss += loss_func(fused_out, yt)
    aln_loss = aln_loss / len(all_rates)
    aln_loss = aln_loss + loss_func(yfs[-1], yt) * 0.1
    
    return aln_loss, fused_out
        


# training
def train():
    
    # init wandb
    if args.wandb:
        wandb.login(key="60fd0a73c2aefc531fa6a3ad0d689e3a4507f51c")
        wandb.init(
            project='EUNAF',
            group=f'x{args.scale}_FINAL',
            name=fname, 
            entity='nmduonggg',
            config=vars(args)
        )
    
    best_perf = -1e9 # psnr
    if args.train_stage > 0:
        core.freeze_backbone()
    
    for epoch in range(epochs):
        track_dict = {}
        
        if epoch % args.val_each == 0:
            perfs_val = [0 for _ in range(num_blocks)]
            total_val_loss = 0.0
            perf_fused = 0.0
            uncertainty = [0 for _ in range(num_blocks)]
            #walk through the test set
            core.eval()
            for m in core.modules():
                if hasattr(m, '_prepare'):
                    m._prepare()
            for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtest), total=len(XYtest)):
                x  = x.cuda()
                yt = yt.cuda()

                with torch.no_grad():
                    out = core.eunaf_forward(x)
                
                outs_mean, masks = out
                perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
                
                if args.train_stage==0:
                    val_loss = loss_func(outs_mean[-1], yt)
                elif args.train_stage==1:
                    val_loss = loss_esu(outs_mean, masks, yt, freeze_mask=False)
                elif args.train_stage==2:
                    # align_biases = core.align_biases
                    val_loss, val_fused = loss_alignment_2(outs_mean, masks, yt)
                    perf_fused += evaluation.calculate(args, val_fused, yt)
                    
                total_val_loss += val_loss.item() if torch.is_tensor(val_loss) else val_loss
                
                for i, p in enumerate(perf_layers_mean):
                    perfs_val[i] += p
                    uncertainty[i] += masks[i].cpu().detach().mean().item()

            perfs_val = [p / len(XYtest) for p in perfs_val]
            perf_fused /= len(XYtest)
            print(perfs_val, perf_fused)
            uncertainty = [u / len(XYtest) for u in uncertainty]
            total_val_loss /= len(XYtest)

            for i, p in enumerate(perfs_val):
                track_dict["val_perf_"+str(i)] = p
                track_dict["val_unc_"+str(i)] = uncertainty[i]
                
            track_dict["val_l1_loss"] = total_val_loss

            log_str = f'[INFO] Epoch {epoch} - Val L: {total_val_loss}'
            print(log_str)
            # torch.save(core.state_dict(), os.path.join(out_dir, f'E_%d_P_%.3f.t7' % (epoch, mean_perf_f)))
            
            if args.train_stage==0:
                if perfs_val[-1] > best_perf:
                    
                    best_perf = perfs_val[-1]
                    torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                    print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))    
            elif args.train_stage==1:
                if torch.tensor(perfs_val).mean() > best_perf:
                    
                    best_perf = torch.tensor(perfs_val).mean()
                    torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                    print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))    
                    
            elif args.train_stage==2:
                if perf_fused > best_perf:
                    best_perf = perf_fused
                    torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                    print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_perf))    
            
            if early_stopper.early_stop(total_val_loss):
                break
        
        # start training 
        total_loss = 0.0
        perfs = [0 for _ in range(num_blocks)]
        uncertainty = [0 for _ in range(num_blocks)]
        
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            
            # intialize
            x  = x.cuda()
            yt = yt.cuda()
            train_loss = 0.0
            
            # inference
            out = core.eunaf_forward(x)   # outs, density, mask
            outs_mean, masks = out
            
            perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
            
            train_loss = 0.0
            if args.train_stage==0:
                # for out_ in outs_mean:
                #     train_loss += loss_func(out_, yt)
                train_loss += loss_func(outs_mean[-1], yt)
                # train_loss /= len(outs_mean)
            elif args.train_stage==1:
                train_loss = loss_esu(outs_mean, masks, yt, freeze_mask=False)
            else:
                # align_biases = core.align_biases
                # train_loss, _ = loss_alignment(outs_mean, masks, yt, align_biases=None, trainable_mask=False)
                train_loss, _ = loss_alignment_2(outs_mean, masks, yt)
            
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() if torch.is_tensor(train_loss) else train_loss
            
            for i, p in enumerate(perf_layers_mean):
                perfs[i] = perfs[i] + p
                uncertainty[i] = uncertainty[i] + (masks[i]).detach().cpu().mean()
        lr_scheduler.step()
        total_loss /= len(XYtrain)
        perfs = [p / len(XYtrain) for p in perfs]
        uncertainty = [u / len(XYtrain) for u in uncertainty]
        
        for i, p in enumerate(perfs):
            track_dict["perf_"+str(i)] = p
        track_dict["train_loss"] = total_loss
        
        log_str = '[INFO] E: %d | LOSS: %.3f | Uncertainty: %.3f' % (epoch, total_loss, uncertainty[-1])
        print(log_str)
        
        if args.wandb: wandb.log(track_dict)
        torch.save(core.state_dict(), os.path.join(out_dir, '_last.t7'))
        
if __name__=='__main__':
    train()