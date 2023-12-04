import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import os
from torchvision.utils import save_image, make_grid

def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# record functions
def record_train(args, epoch,  epoch_loss, epoch_mse, epoch_kld, beta, tfr):
    if args.kl_anneal_cyclical:
        record_txt = "{}/train_record_cyclical.txt".format(args.log_dir)
    else:  
        record_txt = "{}/train_record_monotonic.txt".format(args.log_dir)

    with open(record_txt, "a") as train_record:
        train_record.write(
            ("[epoch: %d] loss: %.5f | mse loss: %.5f | kld loss: %.5f | tf ratio: %.5f | kld beta: %.5f\n" % \
                (
                    epoch,  epoch_loss, epoch_mse, epoch_kld, tfr,  beta
                )
            )
        )

def record_val(val_loss, ave_psnr, best_val_psnr, model, optimizer, args, epoch, beta, tfr):
    if args.kl_anneal_cyclical:
        record_txt = "{}/valid_record_cyclical.txt".format(args.log_dir)
    else:  
        record_txt = "{}/valid_record_monotonic.txt".format(args.log_dir)

    with open(record_txt, 'a') as train_record:
        train_record.write(
            ('[epoch: %d] valid loss: %.5f | ave_psnr = %.5f ===========\n'% \
                (
                    epoch,  val_loss, ave_psnr
                )
            )
        )

    if ave_psnr > best_val_psnr:
        best_val_psnr = ave_psnr
        print("[epoch: %d] best psnr: %.5f" % (epoch, best_val_psnr))
        # save the model
        if args.kl_anneal_cyclical:
            ckpt_dir = os.path.join(args.best_model_dir, 'best_model_cyclical.pt')
        else:
            ckpt_dir = os.path.join(args.best_model_dir, 'best_model_monotonic.pt')
        save_ckpt(ckpt_dir, model, optimizer, args, epoch, beta, tfr, best_val_psnr)

    return best_val_psnr

def save_ckpt(ckpt_dir, model, optimizer, args, epoch, beta, tfr, best_val_psnr):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
        'last_epoch': epoch,
        'beta': beta,
        'tfr': tfr, 
        'best_val_psnr': best_val_psnr
    }

    torch.save(state, ckpt_dir)

def load_ckpt(args, load_ckpt_dir, model, optimizer):
    state = torch.load(load_ckpt_dir)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    start_epoch = state['last_epoch'] + 1
    beta = state['beta']
    tfr = state['tfr']
    
    # load best val psnr from best model
    if args.kl_anneal_cyclical:
        ckpt_dir = os.path.join(args.best_model_dir, 'best_model_cyclical.pt')
    else:
        ckpt_dir = os.path.join(args.best_model_dir, 'best_model_monotonic.pt')

    best_state = torch.load(load_ckpt_dir)
    best_val_psnr = best_state['best_val_psnr']

    print('model loaded from %s' % load_ckpt_dir)
    return model, optimizer, start_epoch, beta, tfr, best_val_psnr

def plot_pred(seq, pred_seq, args, device, sample_idx=0):
    if args.kl_anneal_cyclical:
        gen_dir = "gen/cyclical"
    else:
        gen_dir = "gen/monotonic"
    os.makedirs(gen_dir, exist_ok=True)
	## First one of this batch
    images, pred_frames, gt_frames = [], [], []
    sample_seq, gt_seq = pred_seq[:, sample_idx, :, :, :], seq[:, sample_idx, :, :, :]
    for frame_idx in range(sample_seq.shape[0]):
        gt_frames.append(gt_seq[frame_idx])
        pred_frames.append(sample_seq[frame_idx])

        img_file = "{}/{}.png".format(gen_dir, frame_idx)
        save_image(sample_seq[frame_idx], img_file)
        images.append(imageio.imread(img_file))
        os.remove(img_file)
    
    pred_grid = make_grid(pred_frames, nrow=sample_seq.shape[0])
    gt_grid   = make_grid(gt_frames  , nrow=gt_seq.shape[0])
    save_image(pred_grid, "{}/pred_grid.png".format(gen_dir))
    save_image(gt_grid  , "{}/gt_grid.png".format(gen_dir))
    imageio.mimsave("{}/animation.gif".format(gen_dir), images)