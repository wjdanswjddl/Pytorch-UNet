import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from dice_loss import dice_coeff

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_amp = True


def eval_dice(net, loader, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    tot = 0
    for img, true_mask in tqdm(loader):
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            mask_pred = net(img)
        mask_pred = (mask_pred > 0.5).float()
        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / len(loader)

def eval_loss(net, criterion, loader, gpu=False):
    tot = 0
    for img, true_mask in tqdm(loader):
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            masks_pred = net(img)
        masks_probs_flat = masks_pred.view(-1)
        true_masks_flat = true_mask.view(-1)

        loss = criterion(masks_probs_flat, true_masks_flat)
        tot += loss.item()
    return tot / len(loader)

def eval_dice_loss(net, loader, criterion, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    tot_dice = 0
    tot_loss = 0
    for img, true_mask in tqdm(loader):
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            pred = net(img)
            pred_mask = (pred > 0.5)
            pred_mask = pred_mask.type(torch.float16)
            tot_dice += dice_coeff(pred_mask, true_mask).item()

        pred_flat = pred.view(-1)
        true_mask_flat = true_mask.view(-1)
        loss = criterion(pred_flat, true_mask_flat)
        tot_loss += loss.item()

        dice = tot_dice / len(loader)
        loss = tot_loss / len(loader)
    return dice, loss

def eval_img(net, loader, gpu=False):
    for img, true_mask in tqdm(loader):
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            pred = net(img)
        return true_mask.cpu().numpy(), pred.cpu().numpy()


def eval_roi(f0, f1, th0 = 0, th1 = 0.5):
    '''
    f0 is the denominator
    '''
    f0m = f0.copy()
    f1m = f1.copy()
    f0m[f0m<=th0] = 0
    f0m[f0m>th0] = 1
    f1m[f1m<=th1] = 0
    f1m[f1m>th1] = 1
    num = 0
    den = 0
    for ich in range(0, f0m.shape[1]):
        start = 0
        end = 0
        for it in range(0, f0m.shape[0]):
            if f0m[it,ich] <= 0:
                if start < end:
                    # print(ich, ', ', start, ', ', end, ' : ', np.count_nonzero(f1m[start:end,ich]))
                    den = den + 1
                    if np.count_nonzero(f1m[start:end,ich]) > 0:
                        num = num + 1
                start = it
            else:
                end = it
    if den <= 0:
        return 0
    # print("eval_roi: ", num, "/", den, " = ", (num)/den*100, "%")
    # return [num, den]
    return num/den

def eval_pixel(f0, f1, th0 = 0, th1 = 0.5):
    '''
    f0 is the denominator
    '''
    f0m = f0.copy()
    f1m = f1.copy()
    f0m[f0m<=th0] = 0
    f0m[f0m>th0] = 1
    f1m[f1m<=th1] = 0
    f1m[f1m>th1] = 1
    num = np.count_nonzero(np.logical_and(f0m, f1m))
    den = np.count_nonzero(f0m)
    if den <= 0:
        return 0
    # print("eval_pixel: ", num, "/", den, " = ", (num)/den*100, "%")
    # return [num, den]
    return num/den

def eval_eff_pur(net, dataset, th=0.5, gpu=False):
    eff_pix = 0
    pur_pix = 0
    eff_roi = 0
    pur_roi = 0
    for i, b in enumerate(dataset):
        img = b[0]
        mask_true = b[1]

        if net == "trad":
            # print("Traditional ROI prediction")
            mask_pred = img

        else:
            # print("DNN ROI prediction")
            img = torch.from_numpy(img).unsqueeze(0)
    
            if gpu:
                img = img.cuda()
    
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    mask_pred = net(img).squeeze().cpu().numpy()

        mask_true = np.transpose(mask_true, [1, 0])
        mask_pred = np.transpose(mask_pred, [1, 0])

        eff_pix = eff_pix + eval_pixel(mask_true, mask_pred, 0.5, th)
        pur_pix = pur_pix + eval_pixel(mask_pred, mask_true, th, 0.5)

        eff_roi = eff_roi + eval_roi(mask_true, mask_pred, 0.5, th)
        pur_roi = pur_roi + eval_roi(mask_pred, mask_true, th, 0.5)
    
    eff_pix = eff_pix/(i+1)
    pur_pix = pur_pix/(i+1)
    eff_roi = eff_roi/(i+1)
    pur_roi = pur_roi/(i+1)

    return [eff_pix, pur_pix, eff_roi, pur_roi]
