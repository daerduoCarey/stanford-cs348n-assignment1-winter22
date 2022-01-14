import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from data import ChairDataset
import train_utils as utils
from model import Network


def train(conf):
    # create training and validation datasets and data loaders
    data_features = ['pc', 'shape_id']
    
    train_dataset = ChairDataset(conf.data_dir, 'train', data_features, num_point=conf.num_point)
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, pin_memory=True, \
            num_workers=conf.num_workers, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    
    val_dataset = ChairDataset(conf.data_dir, 'val', data_features, num_point=conf.num_point)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, \
            num_workers=0, drop_last=True, collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)

    # create models
    network = Network(conf)
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    models = [network]
    model_names = ['network']

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    optimizers = [network_opt]
    optimizer_names = ['network_opt']

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    ReconLoss    KLDivLoss   TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # send parameters to device
    for m in models:
        m.to(conf.device)
    for o in optimizers:
        utils.optimizer_to_device(o, conf.device)

    # start training
    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_val_console_log_step = None, None
    train_num_batch = len(train_dataloader)
    val_num_batch = len(val_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)
        train_fraction_done = 0.0
        val_fraction_done = 0.0
        val_batch_ind = -1

        # train for every batch
        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                    step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
                            epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    utils.printout(conf.flog, 'DONE')
                    last_checkpoint_step = train_step

            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind+1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                        val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                for m in models:
                    m.eval()

                with torch.no_grad():
                    # forward pass (including logging)
                    __ = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                            step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                            log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer, lr=network_opt.param_groups[0]['lr'])
           
    # save the final models
    utils.printout(conf.flog, 'Saving final checkpoint ...... ')
    utils.save_checkpoint(models=models, model_names=model_names, dirname=os.path.join(conf.exp_dir, 'ckpts'), \
            epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    utils.printout(conf.flog, 'DONE')


def forward(batch, data_features, network, conf, \
        is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
        log_console=False, log_tb=False, tb_writer=None, lr=None):
    # prepare input
    input_pcs = torch.cat(batch[data_features.index('pc')], dim=0).to(conf.device)           # B x N x 3
    batch_size = input_pcs.shape[0]

    # forward through the network
    output_pcs, pc_feats, ret_list = network(input_pcs)     # B x N x 3, B x P
    
    # for each type of loss, compute losses per data
    recon_loss_per_data = network.get_loss(input_pcs, output_pcs)

    kldiv_loss_per_data = torch.zeros_like(recon_loss_per_data)
    if conf.probabilistic:
        kldiv_loss_per_data = ret_list['kldiv_loss']
    
    # for each type of loss, compute avg loss per batch
    recon_loss = recon_loss_per_data.mean()
    kldiv_loss = kldiv_loss_per_data.mean()

    # compute total loss
    total_loss = conf.cd_loss_weight * recon_loss + \
            conf.kldiv_loss_weight * kldiv_loss

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog, \
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{data_split:^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{recon_loss.item():>10.5f}'''
                f'''{kldiv_loss.item():>10.5f}'''
                f'''{total_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('recon_loss', recon_loss.item(), step)
            tb_writer.add_scalar('kldiv_loss', kldiv_loss.item(), step)
            tb_writer.add_scalar('total_loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)

        # gen visu
        if is_val and (not conf.no_visu) and epoch % conf.num_epoch_every_visu == 0:
            visu_dir = os.path.join(conf.exp_dir, 'val_visu')
            out_dir = os.path.join(visu_dir, 'epoch-%04d' % epoch)
            input_pcs_dir = os.path.join(out_dir, 'input_pcs')
            output_pcs_dir = os.path.join(out_dir, 'output_pcs')
            info_dir = os.path.join(out_dir, 'info')

            if batch_ind == 0:
                # create folders
                os.mkdir(out_dir)
                os.mkdir(input_pcs_dir)
                os.mkdir(output_pcs_dir)
                os.mkdir(info_dir)

            if batch_ind < conf.num_batch_every_visu:
                utils.printout(conf.flog, 'Visualizing ...')

                for i in range(batch_size):
                    fn = 'data-%03d.png' % (batch_ind * batch_size + i)

                    utils.render_pts(os.path.join(input_pcs_dir, fn), input_pcs[i].cpu().numpy())
                    utils.render_pts(os.path.join(output_pcs_dir, fn), output_pcs[i].cpu().numpy())
                    
                    with open(os.path.join(info_dir, fn.replace('.png', '.txt')), 'w') as fout:
                        fout.write('shape_id: %s\n' % batch[data_features.index('shape_id')][i])
                        fout.write('recon_loss: %f\n' % recon_loss_per_data[i].item())
                        fout.write('kldiv_loss: %f\n' % kldiv_loss_per_data[i].item())
                
    return total_loss


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()
    
    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix', default='ae')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    #parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--data_dir', type=str, help='data directory', default='chair_dataset')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--num_point', type=int, default=1024)
    parser.add_argument('--probabilistic', action='store_true', default=False, help='probabilistic [default: False]')

    # training parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)

    # loss weights
    parser.add_argument('--cd_loss_weight', type=float, default=1e3)
    parser.add_argument('--kldiv_loss_weight', type=float, default=1e-3)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='number of optimization steps beween checkpoints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()


    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.exp_suffix}'
    
    # mkdir exp_dir; ask for overwrite if necessary
    conf.exp_dir = conf.exp_name
    if os.path.exists(conf.exp_dir):
        if not conf.overwrite:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.exp_dir)
    os.mkdir(conf.exp_dir)
    os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
    if not conf.no_visu:
        os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # backup python files used for this training
    os.system('cp data.py %s %s' % (__file__, conf.exp_dir))
     
    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    ### start training
    train(conf)


    ### before quit
    # close file log
    flog.close()


