"""
Train EUNAF-version of backbone SISR network
"""

import os
import torch
import numpy as np
import torch.utils.data as torchdata
import torch.nn.functional as F
import torch.nn as nn
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
XYtrain = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)

n_sample = len(trainset)
print('[INFO] trainset contains %d samples' % (n_sample))

# load test data
print('[INFO] load testset "%s" from %s' % (args.testset_tag, args.testset_dir))
testset, batch_size_test = data.load_testset(args)
XYtest = torchdata.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=4)

cost_dict = {
    'srresnet': [0.681, 1.286, 1.890, 5.338],
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
    fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage-1}' if args.n_resgroups > 0 \
        else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage-1}'
    out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', 'Error-predict', fname)
    if os.path.exists(out_dir):
        args.weight = os.path.join(out_dir, '_best.t7')
        print(f"[INFO] Load weight from {args.weight}")
        core.load_state_dict(torch.load(args.weight), strict=True)
    args.weight = '/mnt/disk1/nmduong/FusionNet/Supernet-SR/checkpoints/PRETRAINED/SRResNet/arm-srresnet.pth'
    core.load_state_dict(torch.load(args.weight), strict=True)
    print(f"[INFO] Load weight from {args.weight}")
    
utils.calc_flops(core, (1, 3, 32, 32))
    
core.cuda()

# initialization
lr = args.lr
batch_size = args.batch_size
epochs = args.max_epochs - args.start_epoch
num_blocks = args.n_resgroups if args.n_resgroups > 0 else args.n_resblocks
num_blocks = min(num_blocks//2, args.n_estimators)

ee_params, tail_params = [], []
for n, p in core.named_parameters():
    if p.requires_grad:
        if 'predictors' in n or 'estimators' in n:
            ee_params.append(p)
        else:
            tail_params.append(p)
            

optimizer = Adam([
        {"params": ee_params}, 
        {"params": tail_params, 'lr': 1e-8}
    ], lr=lr, weight_decay=args.weight_decay)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
early_stopper = utils.EarlyStopper(patience=15)
loss_func = loss.create_loss_func(args.loss)

# working dir
fname = name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_ng{args.n_resgroups}_st{args.train_stage}' if args.n_resgroups > 0 \
    else name+f'_x{args.scale}_nb{args.n_resblocks}_nf{args.n_feats}_st{args.train_stage}'
out_dir = os.path.join(args.cv_dir, 'jointly_nofreeze', 'Error-predict', fname)
os.makedirs(out_dir, exist_ok=True)
print("Load ckpoint to: ", out_dir)

def loss_l1s(yfs, yt):
    l1_loss = 0
    for i, yf in enumerate(yfs):
        if i==len(yfs)-1:
            l1_loss += loss_func(yf, yt)
            continue
        l1_loss += loss_func(yf, yt)
    return l1_loss

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

def loss_esu_psnr(yfs, masks, yt, freeze_mask=False):
    assert len(yfs)==len(masks), "yfs contains {%d}, while masks contains {%d}" % (len(yfs), len(masks))
    esu = 0.0
    # mean_mask = torch.mean(torch.cat(masks, dim=0), dim=0)
    # yt = yt.repeat(mean_yf.shape[0], 1, 1, 1)
    ori_yt = yt.clone()
    all_masks = torch.exp(torch.stack(masks, dim=0)).clone().detach()
    pmin = torch.amin(all_masks, dim=0)
    pmax = torch.amax(all_masks, dim=0)
    lbda = 0.1
    
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
        esu = esu + sl1_loss * lbda
        
        psnr = evaluation.calculate(args, yf, yt)
        esu += loss_func(torch.mean(mask_, dim=(1,2,3)), -psnr)
        
    return esu

def loss_esu_psnr_softmax(yfs, masks, yt, freeze_mask=False):
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
        mask_ = masks[i]
        
        s = torch.exp(-mask_)
        yf = torch.mul(yf, s)
        yt = torch.mul(ori_yt, s)
        sl1_loss = loss_func(yf, yt)
        esu = esu + sl1_loss
        
        psnr = evaluation.calculate(args, yf, yt)
    
    all_masks = torch.mean(torch.exp(torch.stack(masks, dim=-1), dim=(1,2,3))) # BxCxHxWxL -> BxL
    psnrs = torch.stack([evaluation.calculate(args, yf, yt) for yf in yfs], dim=1)  # -> BxL
    dist_psnrs = torch.argmax(psnrs, dim=1) # B (label)
    psnr_loss = nn.CrossEntropyLoss()(all_masks, dist_psnrs)
    esu += psnr_loss
    
    return esu

def loss_uncertainty_error(yfs, masks, yt, freeze_mask=False):
    assert len(yfs)==len(masks), "yfs contains {%d}, while masks contains {%d}" % (len(yfs), len(masks))
    esu = 0.0
    # mean_mask = torch.mean(torch.cat(masks, dim=0), dim=0)
    # yt = yt.repeat(mean_yf.shape[0], 1, 1, 1)
    ori_yt = yt.clone()
    all_masks = torch.exp(torch.stack(masks, dim=0)).clone().detach()
    pmin = torch.amin(all_masks, dim=0)
    pmax = torch.amax(all_masks, dim=0)
    
    bce_loss = nn.BCELoss()
    lbda = 2.0
    
    error_loss = 0.0
    for i, yf in enumerate(yfs):
        cur_err = torch.abs(yf - yt).clone().detach()
        pmin = torch.amin(cur_err, dim=(2,3), keepdim=True)
        pmax = torch.amax(cur_err, dim=(2,3), keepdim=True)
        normalized_err = (cur_err - pmin) / (pmax - pmin + 1e-8)
        
        cur_mask = masks[i]
        error_loss += bce_loss(torch.sigmoid(cur_mask), normalized_err) * lbda
    
        yf = yfs[i]
        
        s = torch.exp(-cur_mask)
        yf = torch.mul(yf, s)
        yt = torch.mul(ori_yt, s)
        sl1_loss = loss_func(yf, yt)
        # esu = esu + (2*cur_mask.mean() + sl1_loss)
        esu += sl1_loss
        
    unc_err_loss = esu + error_loss
        
    return unc_err_loss

def loss_error_predict(yfs, masks, yt, freeze_mask=False):
    assert len(yfs)==len(masks), "yfs contains {%d}, while masks contains {%d}" % (len(yfs), len(masks))
    lbda = 2.0
    error_loss = 0.0
    for i, yf in enumerate(yfs):
        cur_err = torch.abs(yf - yt).clone().detach()
        cur_mask = masks[i]
        error_loss += loss_func(cur_mask, cur_err) 
        
    return error_loss

def loss_unc_predict(yfs, masks, yt, freeze_mask=False):
    assert len(yfs)==len(masks), "yfs contains {%d}, while masks contains {%d}" % (len(yfs), len(masks))
    lbda = 2.0
    
    error_loss = 0.0
    for i, yf in enumerate(yfs):
        cur_err = torch.abs(yf.clone().detach() - yt)
        cur_mask = masks[i] * torch.randn_like(masks[i].detach())
        error_loss += loss_func(cur_mask, cur_err) 
        
    return error_loss

def loss_choose_ee(yfs, masks, yt, trainable_mask=False):
    global cost_ees
    all_masks = torch.stack(masks, dim=0).clone().detach()   # L,B,C,H,W
    all_masks_mean = torch.mean(all_masks, dim=(2,3,4)) # L,B
    
    cost_ee = torch.tensor(cost_ees).reshape(-1, 1).to(all_masks.device)
    cost_ee = (cost_ee - cost_ee.min()) / (cost_ee.max() - cost_ee.min() + 1e-8) * all_masks_mean.max()
    
    train_loss = 0.0
    etas = [0.0, 0.3, 0.5]
    
    for eta in etas:
        all_masks_mean_tmp = all_masks_mean.clone().detach()
        all_masks_mean_tmp += (cost_ee * eta)
        
        all_masks_mean_tmp = torch.argmin(all_masks_mean_tmp, dim=0)
        all_masks = F.one_hot(all_masks_mean_tmp, num_classes=len(masks)).int()   # B,L
        bs, num_layers = all_masks.shape
        
        out_patches = torch.zeros_like(yfs[0])
        for i, yf in enumerate(yfs):
            yf = yfs[i] # BCHW
            yf = torch.mul(yf, all_masks[..., i].reshape(bs, 1, 1, 1))
            out_patches += yf
        
        loss = loss_func(out_patches, yt)
        train_loss += loss
    train_loss /= len(etas)
    
    return train_loss

def loss_alignment(yfs, masks, yt, trainable_mask=False):
    
    aln_loss_2 = 0.0
    for i, yf in enumerate(yfs):
    
        final_mask = masks[i].clone().detach() 
        pmin = torch.amin(final_mask, dim=[2,3], keepdim=True)
        pmax = torch.amax(final_mask, dim=[2,3], keepdim=True)
        final_mask = (final_mask - pmin) / (pmax - pmin+ 1e-9)
        
        if i==len(yfs)-1: continue
        yf_ = yf * final_mask
        yt_ = yt * final_mask 
        aln_loss_2 += loss_func(yf_, yt_)
    aln_loss_2 / (len(yfs)-1)
    masks = [
        F.interpolate(torch.mean(torch.exp(m), dim=1, keepdim=True), scale_factor=0.25, mode='bilinear') for m in masks
    ]
    all_masks = torch.stack(masks, dim=-1) # BxCxHxWxN
    all_masks = all_masks.clone().detach()
    raw_indices = torch.argmin(all_masks, dim=-1)    # BxCxHxW
    onehot_indices = F.one_hot(raw_indices, num_classes=len(masks)).float() # Bx1xHxWx4
        
    fused_out = torch.zeros_like(yfs[0])
    for i, yf in enumerate(yfs):
        onehot = F.interpolate(onehot_indices[..., i], scale_factor=4, mode='nearest').int() # Bx1xHxWxN -> Bx1xHxW
        yf = yf * onehot
        fused_out = fused_out + yf
    
    aln_loss = loss_func(fused_out, yt) + aln_loss_2*0.1
    # aln_loss = loss_func(fused_out, yt)
    
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
            group=f'x{args.scale}_UncPsnr_FINAL',
            name=fname, 
            entity='nmduonggg',
            config=vars(args)
        )
    
    step_counter = 0
    best_perf = -1e9 # psnr
    if args.train_stage > 0:
        core.freeze_backbone()
        
        # if args.train_stage > 1:
            # core.enable_estimators_only()
    
    best_val_loss = 1e9
    for epoch in range(epochs):
        track_dict = {}
        
        # start training 
        total_loss = 0.0
        perfs = [0 for _ in range(num_blocks)]
        uncertainty = [0 for _ in range(num_blocks)]
        
        core.train()
        for batch_idx, (x, yt) in tqdm.tqdm(enumerate(XYtrain), total=len(XYtrain)):
            
            ### Start testing ###
            if (step_counter % args.val_each_step == 0):
                print(f"Testing at epoch {epoch} and iterations {step_counter}")
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
                        val_loss = loss_l1s(outs_mean, yt)
                        # val_loss = loss_func(outs_mean[-1], yt)
                    elif args.train_stage==1:
                        val_loss = loss_esu_psnr(outs_mean, masks, yt)
                    elif args.train_stage==2:   # error predict
                        # val_loss = loss_error_predict(outs_mean, masks, yt, freeze_mask=False)
                        val_loss = loss_unc_predict(outs_mean, masks, yt)
                    elif args.train_stage==3:
                        # align_biases = core.align_biases
                        val_loss = loss_choose_ee(outs_mean, masks, yt)
                        # perf_fused += evaluation.calculate(args, val_fused, yt)
                        
                    total_val_loss += val_loss.item() if torch.is_tensor(val_loss) else val_loss
                    
                    for i, p in enumerate(perf_layers_mean):
                        perfs_val[i] += p.mean().item()
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
                    if best_val_loss > total_val_loss:
                        best_val_loss = total_val_loss
                        torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                        print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_val_loss))   
                elif args.train_stage==1:
                    if best_val_loss > total_val_loss:
                        best_val_loss = total_val_loss
                        torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                        print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_val_loss))  
                        
                elif args.train_stage==2:   # error predict
                    if best_val_loss > total_val_loss:
                        best_val_loss = total_val_loss
                        torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                        print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_val_loss))    
                        
                elif args.train_stage==3:   # error-guided early exits
                    if best_val_loss > total_val_loss:
                        best_val_loss = total_val_loss
                        torch.save(core.state_dict(), os.path.join(out_dir, '_best.t7'))
                        print('[INFO] Save best performance model %d with performance %.3f' % (epoch, best_val_loss))  
                
                if args.wandb: wandb.log(track_dict)
                torch.save(core.state_dict(), os.path.join(out_dir, '_last.t7'))
                
                if early_stopper.early_stop(total_val_loss):
                    break
                
            ### End testing ###
            
            # intialize
            x  = x.cuda()
            yt = yt.cuda()
            train_loss = 0.0
            
            # inference
            out = core.eunaf_forward(x)   # outs, density, mask
            outs_mean, masks = out
            
            # freeze batch norm
            for m in core.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
            
            perf_layers_mean = [evaluation.calculate(args, yf, yt) for yf in outs_mean]
            
            train_loss = 0.0
            if args.train_stage==0:
                # for out_ in outs_mean:
                #     train_loss += loss_func(out_, yt)
                train_loss += loss_l1s(outs_mean, yt)
            elif args.train_stage==1:
                train_loss = loss_esu_psnr(outs_mean, masks, yt)
            elif args.train_stage==2:   # error predict
                train_loss = loss_unc_predict(outs_mean, masks, yt, freeze_mask=False)
            elif args.train_stage==3:
                # align_biases = core.align_biases
                # train_loss, _ = loss_alignment(outs_mean, masks, yt, align_biases=None, trainable_mask=False)
                train_loss += loss_choose_ee(outs_mean, masks, yt)
            
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() if torch.is_tensor(train_loss) else train_loss
            
            for i, p in enumerate(perf_layers_mean):
                perfs[i] = perfs[i] + p.mean().item()
                uncertainty[i] = uncertainty[i] + (masks[i]).detach().cpu().mean()
                
            step_counter += 1
                
                
        lr_scheduler.step()
        total_loss /= len(XYtrain)
        perfs = [p / len(XYtrain) for p in perfs]
        uncertainty = [u / len(XYtrain) for u in uncertainty]
        
        for i, p in enumerate(perfs):
            track_dict["perf_"+str(i)] = p
        track_dict["train_loss"] = total_loss
        
        log_str = '[INFO] E: %d | LOSS: %.3f | Uncertainty: %.3f' % (epoch, total_loss, uncertainty[-1])
        print(log_str)
        
        
if __name__=='__main__':
    train()