import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from Scheduler import GradualWarmupScheduler
from dataloader_cifar import load_data, transback
from dataloader_cityscapes import CustomDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size

def train(params:argparse.Namespace):
    # assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # initialize settings
    # init_process_group(backend="nccl")
    # get local rank for each process
    # local_rank = get_rank()
    # set device
    device = torch.device("cuda")
    # load data
    # dataloader = load_data(params.batchsize, params.numworkers)
    train_dataset = CustomDataset('cityscapes_ori/train')
    train_dataloader = DataLoader(train_dataset, batch_size=params.batchsize, shuffle=True)
    valid_dataset = CustomDataset('cityscapes_ori/val')
    valid_dataloader = DataLoader(valid_dataset, batch_size=params.batchsize, shuffle=False)
    # initialize models
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量为：{total_params}")
    cemblayer = ConditionalEmbedding(34).to(device)
    total_params = sum(p.numel() for p in cemblayer.parameters())
    print(f"模型参数量为：{total_params}")
    # load last epoch
    lastpath = os.path.join(params.moddir,'last_epoch.pt')
    if os.path.exists(lastpath):
        lastepc = torch.load(lastpath)['last_epoch']
        # load checkpoints
        checkpoint = torch.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v,
                    device = device
                )
    
    # DDP settings 
    # diffusion.model = DDP(
    #                         diffusion.model,
    #                         device_ids = [local_rank],
    #                         output_device = local_rank
    #                     )
    # cemblayer = DDP(
    #                 cemblayer,
    #                 device_ids = [local_rank],
    #                 output_device = local_rank
    #             )
    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch,
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier = params.multiplier,
                            warm_epoch = params.epoch // 10,
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
    # training
    cnt = torch.cuda.device_count()
    for epc in range(lastepc, params.epoch):
        # turn into train mode
        diffusion.model.train()
        cemblayer.train()
        # sampler.set_epoch(epc)
        # batch iterations
        with tqdm(train_dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for lab, img in tqdmDataLoader:
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                lab = lab.to(device)
                input_label = torch.zeros(b, 34, 64, 64, dtype=torch.float32, device='cuda')
                input_semantics = input_label.scatter_(1, lab, 1.0)
                cemb = cemblayer(input_semantics)
                # cemb[np.where(np.random.rand(b)<params.threshold)] = 0
                loss = diffusion.trainloss(x_0, cemb = cemb)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
        warmUpScheduler.step()
        # evaluation and save checkpoint
        if (epc + 1) % params.interval == 0:
            diffusion.model.eval()
            cemblayer.eval()
            # generating samples
            # The model generate 80 pictures(8 per row) each time
            # pictures of same row belong to the same class
            all_samples = []
            each_device_batch = params.genbatch // cnt
            with torch.no_grad():
                for lab, image in valid_dataloader:
                    image = image.to(device)
                    image = transback(image)
                    lab = lab.to(device)
                    input_label = torch.zeros(b, 34, 64, 64, dtype=torch.float32, device='cuda')
                    input_semantics = input_label.scatter_(1, lab, 1.0)
                    cemb = cemblayer(input_semantics)
                    genshape = (each_device_batch , 3, 128, 64)
                    if params.ddim:
                        generated = diffusion.ddim_sample(genshape, params.num_steps, params.eta, params.select, cemb = cemb)
                    else:
                        generated = diffusion.sample(genshape, cemb = cemb)
                    img = transback(generated)
                    # print(img.shape)
                    img = torch.concat((image, img), dim = 0)
                    img = img.reshape(params.genbatch, 2, 3, 64, 64).contiguous()
                    
                    all_samples = [img.clone() for _ in range(torch.cuda.device_count())]
                    samples = torch.concat(all_samples, dim = 1).reshape(params.genbatch * 2, 3, 64, 64)
                    # if local_rank == 0:
                    save_image(samples, os.path.join(params.samdir, f'generated_{epc+1}_pict.png'), nrow = params.genbatch)
                    break
            # save checkpoints
            checkpoint = {
                                'net':diffusion.model.state_dict(),
                                'cemblayer':cemblayer.state_dict(),
                                'optimizer':optimizer.state_dict(),
                                'scheduler':warmUpScheduler.state_dict()
                            }
            torch.save({'last_epoch':epc+1}, os.path.join(params.moddir,'last_epoch.pt'))
            torch.save(checkpoint, os.path.join(params.moddir, f'ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()
    destroy_process_group()

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=8,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=torch.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=10,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=8,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()

