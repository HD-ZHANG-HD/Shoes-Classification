import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from model.backbone.ResNet import *
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from torch import nn
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import yaml

def train(train_loader, model, optimizer, epoch, save_path, writer):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, ((images_l, cls),(images_w,_),(images_s)) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            cls = cls.cuda()

            preds = model(images)
            loss = criterion_u(preds, cls)


            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                    format(epoch, opt.epoch, i, total_step, loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_total': loss.data},
                                   global_step=step)


        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_loss, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, cls = test_loader.load_data()
            cls = np.asarray(cls, np.float32)
            image = image.cuda()

            res = model(image)

            loss_sum += criterion_u(res, cls)
        loss_ave = loss_sum / test_loader.size
        writer.add_scalar('Loss_Average', torch.tensor(loss_ave), global_step=epoch)
        print('Epoch: {}, Loss Average: {}, best Loss: {}, bestEpoch: {}.'.format(epoch, loss_ave, best_loss, best_epoch))
        if epoch == 1:
            best_loss = loss_ave
        else:
            if loss_ave < best_loss:
                best_loss = loss_ave
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, loss_ave, best_epoch, best_loss))






if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/shoes.yaml')
    opt = parser.parse_args()

    cfg = yaml.load(open(opt.config, "r"), Loader=yaml.Loader)

    # set the device for training
    if cfg['gpu_id'] == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif cfg['gpu_id'] == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
 
    cudnn.benchmark = True

    # build the model
    model = resnet50()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # device_ids = [0, 1]
    # model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()

    if cfg['load'] is not None:
        model.load_state_dict(torch.load(cfg['load']))
        print('load model from ', cfg['load'])

    optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])

    criterion_u = nn.BCEWithLogitsLoss(reduction='none').cuda()

    save_path = cfg['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader_l, train_loader_u = get_loader(cfg=cfg)
    val_loader = test_dataset(image_root=cfg['val_root'],
                              testsize=cfg['trainsize'])
    total_step = len(train_loader_u) * cfg['epoch']

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(cfg['epoch'], cfg['lr'], cfg['batchsize'], cfg['trainsize'], cfg['clip'],
                                                         cfg['decay_rate'], cfg['load'], save_path, cfg['decay_epoch']))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_loss = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, cfg['epoch']):
        cur_lr = adjust_lr(optimizer, cfg['lr'], epoch,
                           cfg['decay_rate'], cfg['decay_epoch'])
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train_loader = zip(train_loader_l, train_loader_u, train_loader_u)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)
