import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import params
from model import net_G, net_D
from utils import ConditionalNpyDataset, generateZ, SavePloat_Voxels



# Morphometric losses


def mdi_ratio_loss(fake, y_norm, y_mean, y_std,
                   vol_idx, height_idx,
                   voxel_size, thresh=0.7, eps=1e-8):
   
    device = fake.device
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    fake_prob = 0.5 * (fake + 1.0)

    # occupancy mask for volume & height
    occ = (fake_prob > thresh).float()  

    # predicted volume 
    n_vox = occ.sum(dim=(2, 3, 4)).squeeze(1)  
    voxel_vol = voxel_size ** 3
    vol_pred_phys = n_vox * voxel_vol         

    # predicted height
    profile_z = occ.sum(dim=(1, 3, 4))       
    mask = (profile_z > 0).float()          
    has_any = (profile_z > 0).any(dim=1)      

    bottom_idx = torch.argmax(mask, dim=1)   
    mask_rev = mask.flip(dims=[1])
    top_from_end = torch.argmax(mask_rev, dim=1)
    D = mask.shape[1]
    top_idx = (D - 1) - top_from_end        

    height_vox = (top_idx - bottom_idx + 1).float()
    height_vox = height_vox * has_any.float()  
    h_pred_phys = height_vox * voxel_size     

    # true physical volume & height from normalized y
    vol_true_norm = y_norm[:, vol_idx]
    h_true_norm   = y_norm[:, height_idx]

    vol_mean = y_mean[vol_idx]
    vol_std  = y_std[vol_idx] + eps
    h_mean   = y_mean[height_idx]
    h_std    = y_std[height_idx] + eps

    vol_true_phys = vol_true_norm * vol_std + vol_mean
    h_true_phys   = h_true_norm   * h_std   + h_mean

    # ratio 
    ratio_pred = h_pred_phys / (vol_pred_phys + eps)
    ratio_true = h_true_phys / (vol_true_phys + eps)

    ratio_pred = ratio_pred.view(-1, 1)
    ratio_true = ratio_true.view(-1, 1)

    return F.l1_loss(ratio_pred, ratio_true)


def mdi_height_loss_z(fake, y_norm, y_mean, y_std,
                      height_idx, voxel_size, thresh=0.7, eps=1e-8):
 
    device = fake.device
    y_mean = y_mean.to(device)
    y_std  = y_std.to(device)

    fake_prob = 0.5 * (fake + 1.0)
    occ = (fake_prob > thresh).float()       
    profile_z = occ.sum(dim=(1, 3, 4))       
    mask = (profile_z > 0).float()
    has_any = (profile_z > 0).any(dim=1)

    bottom_idx = torch.argmax(mask, dim=1)
    mask_rev = mask.flip(dims=[1])
    top_from_end = torch.argmax(mask_rev, dim=1)
    D = mask.shape[1]
    top_idx = (D - 1) - top_from_end

    height_vox = (top_idx - bottom_idx + 1).float()
    height_vox = height_vox * has_any.float()
    height_phys = height_vox * voxel_size    
    h_mean = y_mean[height_idx]
    h_std  = y_std[height_idx] + eps
    height_norm = (height_phys - h_mean) / h_std

    h_true_norm = y_norm[:, height_idx]

    height_norm = height_norm.view(-1, 1)
    h_true_norm = h_true_norm.view(-1, 1)

    return F.mse_loss(height_norm, h_true_norm)


# Trainer


def trainer(args):
    print("=== cGAN + MDI training (hinge loss) ===")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("Using device:", params.device)

    # Paths 
    model_root = os.path.join(params.output_dir, args.model_name)
    model_dir  = os.path.join(model_root, "models")
    image_dir  = os.path.join(model_root, "images")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    npz_history_path = os.path.join(model_root, "loss_history.npz")
    csv_history_path = os.path.join(model_root, "training_history.csv")

    # Dataset / DataLoader 
    train_ds_tmp = ConditionalNpyDataset(
        params.train_npy_dir,
        params.train_cond_csv
    )
    y_mean, y_std = train_ds_tmp.y_mean, train_ds_tmp.y_std
    params.cond_dim = int(train_ds_tmp.cond_dim)

    # saving normalization stats
    np.savez(
        os.path.join(model_root, params.cond_norm_file),
        mean=y_mean,
        std=y_std,
        cond_dim=params.cond_dim,
    )

    train_dataset = ConditionalNpyDataset(
        params.train_npy_dir,
        params.train_cond_csv,
        y_mean=y_mean,
        y_std=y_std
    )

    loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    y_mean_t = torch.from_numpy(train_dataset.y_mean).to(params.device)
    y_std_t  = torch.from_numpy(train_dataset.y_std).to(params.device)
    vol_idx  = params.volume_index
    ht_idx   = params.height_index

    # Models 
    G = net_G(args).to(params.device)
    D = net_D(args).to(params.device)
    print("G on:", next(G.parameters()).device)
    print("D on:", next(D.parameters()).device)

    # Optimizers, schedulers 
    d_weight_decay = getattr(params, "d_weight_decay", 0.0)

    opt_G = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)
    opt_D = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta,
                       weight_decay=d_weight_decay)

    sched_G = optim.lr_scheduler.StepLR(
        opt_G, step_size=params.lr_step, gamma=params.lr_gamma
    )
    sched_D = optim.lr_scheduler.StepLR(
        opt_D, step_size=params.lr_step, gamma=params.lr_gamma
    )

    # Resume checkpoint & history 
    start_epoch = 300
    checkpoint_path = os.path.join(model_dir, "checkpoint_latest.pth")

    epoch_D_losses       = []
    epoch_G_losses       = []
    epoch_G_adv_losses   = []
    epoch_G_morph_losses = []
    lr_G_hist            = []
    lr_D_hist            = []

    if getattr(params, "resume_training", True) and os.path.exists(checkpoint_path):
        print(f"==> Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=params.device)

        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        if "sched_G" in ckpt:
            sched_G.load_state_dict(ckpt["sched_G"])
        if "sched_D" in ckpt:
            sched_D.load_state_dict(ckpt["sched_D"])

        start_epoch = ckpt["epoch"] + 1
        print(f"==> Resuming at epoch {start_epoch}")

        # load old curves if present
        if os.path.exists(npz_history_path):
            h = np.load(npz_history_path)
            keys = h.files

            epoch_D_losses = h["D"].tolist()
            epoch_G_losses = h["G"].tolist()
            lr_G_hist      = h["lr_G"].tolist()
            lr_D_hist      = h["lr_D"].tolist()

            if "G_adv" in keys:
                epoch_G_adv_losses = h["G_adv"].tolist()
            else:
                epoch_G_adv_losses = epoch_G_losses.copy()

            if "G_morph" in keys:
                epoch_G_morph_losses = h["G_morph"].tolist()
            else:
                epoch_G_morph_losses = [0.0] * len(epoch_G_losses)

            L = min(start_epoch, len(epoch_D_losses))
            epoch_D_losses       = epoch_D_losses[:L]
            epoch_G_losses       = epoch_G_losses[:L]
            epoch_G_adv_losses   = epoch_G_adv_losses[:L]
            epoch_G_morph_losses = epoch_G_morph_losses[:L]
            lr_G_hist            = lr_G_hist[:L]
            lr_D_hist            = lr_D_hist[:L]
    else:
        print("==> No checkpoint found or resume disabled. Starting from epoch 0.")
    # manual restart override (forces start at a specific epoch) 
    manual_restart = getattr(params, "manual_restart_epoch", None)

    if manual_restart is not None:
        start_epoch = manual_restart
        print(f"==> Forcing restart from epoch {start_epoch} using G/D_epoch_{start_epoch}.pth")

        g_path = os.path.join(model_dir, f"G_epoch_{start_epoch}.pth")
        d_path = os.path.join(model_dir, f"D_epoch_{start_epoch}.pth")

        if not (os.path.exists(g_path) and os.path.exists(d_path)):
            raise FileNotFoundError(
                f"Expected {g_path} and {d_path} to exist for manual restart, "
                "but one or both are missing."
            )

        # Load weights into G and D
        G.load_state_dict(torch.load(g_path, map_location=params.device))
        D.load_state_dict(torch.load(d_path, map_location=params.device))

        # Re-init optimizers & schedulers from scratch (fresh state at epoch=start_epoch)
        opt_G = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)
        opt_D = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta,
                        weight_decay=d_weight_decay)
        sched_G = optim.lr_scheduler.StepLR(
            opt_G, step_size=params.lr_step, gamma=params.lr_gamma
        )
        sched_D = optim.lr_scheduler.StepLR(
            opt_D, step_size=params.lr_step, gamma=params.lr_gamma
        )

        # trim loss history up to this epoch
        if os.path.exists(npz_history_path):
            h = np.load(npz_history_path)
            epoch_D_losses       = h["D"].tolist()
            epoch_G_losses       = h["G"].tolist()
            epoch_G_adv_losses   = h["G_adv"].tolist()
            epoch_G_morph_losses = h["G_morph"].tolist()
            lr_G_hist            = h["lr_G"].tolist()
            lr_D_hist            = h["lr_D"].tolist()

            L = start_epoch
            epoch_D_losses       = epoch_D_losses[:L]
            epoch_G_losses       = epoch_G_losses[:L]
            epoch_G_adv_losses   = epoch_G_adv_losses[:L]
            epoch_G_morph_losses = epoch_G_morph_losses[:L]
            lr_G_hist            = lr_G_hist[:L]
            lr_D_hist            = lr_D_hist[:L]

            print(f"==> Trimmed loss history to first {L} epochs for plotting.")



    # CSV history 
    append_mode = os.path.exists(csv_history_path) and (start_epoch > 0)
    history_file = open(csv_history_path, "a" if append_mode else "w", newline="")
    history_writer = csv.writer(history_file)

    if not append_mode:
        history_writer.writerow(
            ["epoch", "D_loss", "G_loss", "G_adv_loss", "G_morph_loss", "lr_G", "lr_D"]
        )
        history_file.flush()



    #  Morph loss weight
    morph_weight_ratio = getattr(params, "morph_weight_vol", 1e-4)
    morph_weight_ht    = getattr(params, "morph_weight_ht", 1e-4)

    last_fake_batch = None


    #  TRAIN LOOP
    
    for epoch in range(start_epoch, params.epochs):
        G.train()
        D.train()

        running_D       = 0.0
        running_G_total = 0.0
        running_G_adv   = 0.0
        running_G_morph = 0.0

        for X, y in tqdm(loader, desc=f"Epoch {epoch+1}/{params.epochs}"):
            X = X.to(params.device)
            y = y.to(params.device)
            B = X.size(0)

            real_vol = X.unsqueeze(1)

            #Train D 
            Z = generateZ(args, B)
            fake = G(Z, y)

            # small noise on conditions for D
            cond_noise_std = 0.05
            y_d = y + cond_noise_std * torch.randn_like(y)
            y_d = torch.clamp(y_d, -3.0, 3.0)

            real_logits_D = D(real_vol, y_d)
            fake_logits_D = D(fake.detach(), y_d)

            loss_D_real = torch.mean(F.relu(1.0 - real_logits_D))
            loss_D_fake = torch.mean(F.relu(1.0 + fake_logits_D))
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G 
            for _ in range(2):
                Z = generateZ(args, B)
                fake_G = G(Z, y)
                last_fake_batch = fake_G.detach().cpu()

                fake_logits_D_for_G = D(fake_G, y)
                loss_G_adv = -torch.mean(fake_logits_D_for_G)

                loss_G_morph_ratio = mdi_ratio_loss(
                    fake_G, y, y_mean_t, y_std_t,
                    vol_idx, ht_idx, params.voxel_size,
                    thresh=0.7, eps=1e-8
                )

                loss_G_morph_ht = mdi_height_loss_z(
                    fake_G, y, y_mean_t, y_std_t,
                    ht_idx, params.voxel_size,
                    thresh=0.7, eps=1e-8
                )

                loss_G_morph = (
                    morph_weight_ratio * loss_G_morph_ratio +
                    morph_weight_ht    * loss_G_morph_ht
                )

                loss_G = loss_G_adv + loss_G_morph

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            # accumulate
            running_D       += loss_D.item()       * B
            running_G_total += loss_G.item()       * B
            running_G_adv   += loss_G_adv.item()   * B
            running_G_morph += loss_G_morph.item() * B

        N = len(train_dataset)
        epoch_D = running_D       / N
        epoch_G = running_G_total / N
        epoch_G_adv_epoch   = running_G_adv   / N
        epoch_G_morph_epoch = running_G_morph / N

        epoch_D_losses.append(epoch_D)
        epoch_G_losses.append(epoch_G)
        epoch_G_adv_losses.append(epoch_G_adv_epoch)
        epoch_G_morph_losses.append(epoch_G_morph_epoch)

        lr_G_hist.append(opt_G.param_groups[0]['lr'])
        lr_D_hist.append(opt_D.param_groups[0]['lr'])

        print(
            f"Epoch {epoch+1}: "
            f"D_loss={epoch_D:.4f}, "
            f"G_total={epoch_G:.4f}, "
            f"G_adv={epoch_G_adv_epoch:.4f}, "
            f"G_morph={epoch_G_morph_epoch:.4f}, "
            f"lr_G={lr_G_hist[-1]:.6f}, lr_D={lr_D_hist[-1]:.6f}"
        )

        global_epoch = epoch + 1

        history_writer.writerow([
            global_epoch,
            epoch_D,
            epoch_G,
            epoch_G_adv_epoch,
            epoch_G_morph_epoch,
            lr_G_hist[-1],
            lr_D_hist[-1],])

        history_file.flush()

        # step schedulers
        sched_G.step()
        sched_D.step()

        # Saving models & samples 
        if (epoch + 1) % params.model_save_step == 0:
            torch.save(G.state_dict(), os.path.join(model_dir, f"G_epoch_{epoch+1}.pth"))
            torch.save(D.state_dict(), os.path.join(model_dir, f"D_epoch_{epoch+1}.pth"))

            checkpoint = {
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "sched_G": sched_G.state_dict(),
                "sched_D": sched_D.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

            if last_fake_batch is not None:
                fake_np = last_fake_batch[:8].squeeze(1).numpy()
                fake_01 = 0.5 * (fake_np + 1.0)
                SavePloat_Voxels(fake_01, image_dir, f"epoch_{epoch+1}")

        # Saving NPZ history after each epoch
        np.savez(
            npz_history_path,
            D=np.array(epoch_D_losses, dtype=np.float32),
            G=np.array(epoch_G_losses, dtype=np.float32),
            G_adv=np.array(epoch_G_adv_losses, dtype=np.float32),
            G_morph=np.array(epoch_G_morph_losses, dtype=np.float32),
            lr_G=np.array(lr_G_hist, dtype=np.float32),
            lr_D=np.array(lr_D_hist, dtype=np.float32),
        )

    # Final curves 
    epochs = range(1, len(epoch_D_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_D_losses, label="D loss")
    plt.plot(epochs, epoch_G_losses, label="G total loss")
    plt.plot(epochs, epoch_G_adv_losses, label="G adv loss", alpha=0.7, linestyle="--")
    plt.plot(epochs, epoch_G_morph_losses, label="G morph loss", alpha=0.7, linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MDI-cGAN Training Loss")
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(model_root, "loss_curve.png")
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to: {curve_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr_G_hist, label="G learning rate")
    plt.plot(epochs, lr_D_hist, label="D learning rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    lr_curve_path = os.path.join(model_root, "lr_curve.png")
    plt.savefig(lr_curve_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved LR curve to: {lr_curve_path}")

    history_file.close()
