
import torch
cube_len   = 64
z_dim      = 50
z_dis = "norm"
batch_size = 100
# learning rates
g_lr = 2e-4
d_lr = 4e-5


beta = (0.0, 0.9)

epochs = 1000
# loss weights

model_save_step = 50
# device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

checkpoint_file = "checkpoint_latest.pth"
lr_step    = 200
lr_gamma   = 0.5

resume_training = True
cond_dim = 0
cond_norm_file = "cond_stats.npz"
d_weight_decay = 5e-1
morph_weight_ratio = 2e-1
morph_weight_ht =  2e-1
volume_index      = 0
occ_index  = 13
height_index  = 12
voxel_size        = 3.5
manual_restart_epoch = 850



train_npy_dir   = r"/data/imam/project_data/train/"
train_cond_csv = r"/data/imam/project_data/train_final.csv"
output_dir = r"/data/imam/hinge_output/"
seed       = 123
