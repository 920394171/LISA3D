import os
import sys
import logging
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='synapse')
parser.add_argument('--exp', type=str, default='/synapse_20p/lisa3d/fold1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_20p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_20p')
parser.add_argument('-se', '--split_eval', type=str, default='eval')
parser.add_argument('-m', '--mixed_precision', action='store_true', default=True)
parser.add_argument('-ep', '--max_epoch', type=int, default=1500)
parser.add_argument('--cps_loss', type=str, default='w_ce+dice')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.3)
parser.add_argument('-s', '--ema_w', type=float, default=0.99)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=10)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=None)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Import our LISA3D model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from MyModel.model.LISA3D import LISA3DForSegmentation

# Import utilities from SKCDF
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SKCDF')))
from code.utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr
from code.utils.loss import DC_and_CE_loss, RobustCrossEntropyLoss, SoftDiceLoss
from code.data.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_X, RandomFlip_Y
from code.data.data_loaders import Synapse_AMOS
from code.utils.config import Config

config = Config(args.task)

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if args.consistency_rampup is None:
            args.consistency_rampup = args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.cps_w

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def make_loss_function(name, weight=None):
    if name == 'ce':
        return RobustCrossEntropyLoss()
    elif name == 'wce':
        return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice':
        return DC_and_CE_loss()
    elif name == 'wce+dice':
        return DC_and_CE_loss(w_ce=weight)
    elif name == 'w_ce+dice':
        return DC_and_CE_loss(w_dc=weight, w_ce=weight)
    elif name == 'dice':
        return SoftDiceLoss()
    else:
        raise ValueError(name)

def make_loader(split, dst_cls=Synapse_AMOS, repeat=None, is_training=True, unlabeled=False):
    if is_training:
        dst = dst_cls(
            task=args.task,
            split=split,
            repeat=repeat,
            unlabeled=unlabeled,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                RandomCrop(config.patch_size),
                RandomFlip_X(),
                RandomFlip_Y(),
                ToTensor()
            ])
        )
        return DataLoader(
            dst,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=True
        )
    else:
        dst = dst_cls(
            task=args.task,
            split=split,
            is_val=True,
            num_cls=config.num_cls,
            transform=transforms.Compose([
                CenterCrop(config.patch_size),
                ToTensor()
            ])
        )
        return DataLoader(dst, pin_memory=True)

def make_lisa3d_model():
    # Create model configuration
    class ModelConfig:
        def __init__(self):
            self.train_mask_decoder = True
            self.out_dim = 256
            self.hidden_size = 768
            self.n_channels = config.num_channels
            self.num_cls = config.num_cls
    
    model_config = ModelConfig()
    
    # Create model
    model = LISA3DForSegmentation(
        config=model_config,
        ce_loss_weight=1.0,
        dice_loss_weight=1.0,
        bce_loss_weight=1.0,
    ).cuda()
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=3e-5,
        nesterov=True
    )
    return model, optimizer

# Register forward hook to print intermediate shapes for debugging
def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            print(f"[HOOK] {name}: output shape {output.shape}")
        elif isinstance(output, (list, tuple)):
            shapes = []
            for o in output:
                if isinstance(o, torch.Tensor):
                    shapes.append(o.shape)
            print(f"[HOOK] {name}: output shapes {shapes}")
    return hook

def register_hooks(model):
    mod = model.module if hasattr(model, "module") else model
    for name, module in mod.named_modules():
        # Uncomment to enable shape debugging
        # module.register_forward_hook(get_hook(name))
        pass

if __name__ == '__main__':
    import random

    # Set random seeds for reproducibility
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Create log directory
    snapshot_path = f'/data/project/MyModel/logs/{args.exp}/'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))

    # Setup logger
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Create data loaders
    unlabeled_loader = make_loader(args.split_unlabeled, unlabeled=True)
    labeled_loader = make_loader(args.split_labeled, repeat=len(unlabeled_loader.dataset))
    eval_loader = make_loader(args.split_eval, is_training=False)

    logging.info(f'{len(labeled_loader)} iterations per epoch (labeled)')
    logging.info(f'{len(unlabeled_loader)} iterations per epoch (unlabeled)')

    # Create model and optimizer
    model, optimizer = make_lisa3d_model()
    model = xavier_normal_init_weight(model)

    # Register hooks for debugging (if needed)
    # register_hooks(model)

    logging.info(optimizer)

    # Create loss functions
    sup_loss_func = make_loss_function(args.sup_loss)
    unsup_loss_func = make_loss_function(args.cps_loss)

    # Calculate class weights for balanced loss
    num_cls = config.num_cls
    num_sample = 4
    training_set = Synapse_AMOS(split=args.split_labeled, num_cls=num_cls, task=args.task)
    loop = 0
    class_size = [0 for c in range(num_cls)]
    for sample in training_set:
        image = sample["image"]
        label = sample["label"]
        for i in range(num_cls):
            num = np.sum(label == i)
            class_size[i] = class_size[i] + num
        loop = loop + 1
        if loop == num_sample:
            break
    
    ir2 = min(class_size) / np.array(class_size)
    ir2 = torch.tensor(ir2).cuda()

    # Initial consistency weight
    cps_w = get_current_consistency_weight(0)

    # Training tracking variables
    best_eval = 0.0
    best_epoch = 0

    # Training loop
    for epoch_num in range(args.max_epoch + 1):
        loss_list = []
        loss_labeled_list = []
        loss_unlabeled_list = []

        model.train()

        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader)):
            optimizer.zero_grad()
            
            # Get data
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            
            # Convert labels to one-hot format for loss calculation
            masks_list = []
            label_list = []
            for i in range(image_l.shape[0]):
                masks_list.append(label_l[i].unsqueeze(0))
                label_list.append(label_l[i].shape)
            
            # Forward pass
            outputs = model(
                images_labeled=image_l,
                images_unlabeled=image_u,
                masks_list=masks_list,
                label_list=label_list
            )
            
            # Get losses
            loss = outputs["loss"]
            loss_labeled = outputs["loss_labeled"]
            loss_unlabeled = outputs["loss_unlabeled"] * cps_w
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Record losses
            loss_list.append(loss.item())
            loss_labeled_list.append(loss_labeled.item())
            loss_unlabeled_list.append(loss_unlabeled.item())
        
        # Get current learning rate
        lr = get_lr(optimizer)
        
        # Log metrics
        writer.add_scalar('lr', lr, epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/loss', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/labeled', np.mean(loss_labeled_list), epoch_num)
        writer.add_scalar('loss/unlabeled', np.mean(loss_unlabeled_list), epoch_num)
        
        logging.info(
            f'epoch {epoch_num} : loss : {np.mean(loss_list)}, cpsw:{cps_w} lr: {lr} '
        )
        
        # Update learning rate and consistency weight
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        cps_w = get_current_consistency_weight(epoch_num)
        
        # Evaluation
        if epoch_num % 1 == 0:
            dice_list = [[] for _ in range(config.num_cls - 1)]
            model.eval()
            
            dice_func = SoftDiceLoss(smooth=1e-8)
            for batch in tqdm(eval_loader):
                with torch.no_grad():
                    image, gt = fetch_data(batch)
                    # For evaluation, we use the same image for both labeled and unlabeled
                    outputs = model(
                        images_labeled=image,
                        images_unlabeled=image,
                        inference=True
                    )
                    
                    # Use the output from the unlabeled decoder for evaluation
                    output = outputs["output_unlabeled"]
                    
                    shp = output.shape
                    gt = gt.long()
                    y_onehot = torch.zeros(shp).cuda()
                    y_onehot.scatter_(1, gt, 1)
                    
                    x_onehot = torch.zeros(shp).cuda()
                    output = torch.argmax(output, dim=1, keepdim=True).long()
                    x_onehot.scatter_(1, output, 1)
                    
                    dice = dice_func(x_onehot, y_onehot, is_training=False)
                    dice = dice.data.cpu().numpy()
                    for i, d in enumerate(dice):
                        dice_list[i].append(d)
            
            dice_mean = []
            for dice in dice_list:
                dice_mean.append(np.mean(dice))
            
            logging.info(f'evaluation epoch {epoch_num}, dice: {np.mean(dice_mean)}, {dice_mean}')
            
            # Save best model
            if np.mean(dice_mean) > best_eval:
                best_eval = np.mean(dice_mean)
                best_epoch = epoch_num
                save_path = os.path.join(snapshot_path, f'ckpts/best_model.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'saving best model to {save_path}')
            
            logging.info(f'\t best eval dice is {best_eval} in epoch {best_epoch}')
            
            # Early stopping
            if epoch_num - best_epoch == config.early_stop_patience:
                logging.info(f'Early stop.')
                break
    
    writer.close()
