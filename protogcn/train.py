import os
import os.path as osp
import time 
import random
import argparse
import logging
import math
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim


from .datasets import build_dataloader, PoseDataset
from .models import Recognizer
from .models import ProtoGCN
from .models import Head
from .utils import get_logger, dump_file


def parse_args():
    parser = argparse.ArgumentParser(description='Train ProtoGCN')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--test-last', action='store_true')
    parser.add_argument('--test-best', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--load-from', type=str, default=None)
    parser.add_argument('--work-dir', type=str, default=None)
    parser.add_argument('--compile', action='store_true')
    return parser.parse_args()


def load_config(config_path):
    """Load config from Python file"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return {k: getattr(config, k) for k in dir(config) if not k.startswith('_')}


def set_random_seed(seed, deterministic=False):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_losses(losses):
    """Parse losses dict to (loss_tensor, log_vars_dict)."""
    log_vars = OrderedDict()
    for name, value in losses.items():
        log_vars[name] = value.mean() if isinstance(value, torch.Tensor) else sum(v.mean() for v in value)

    loss = sum(v for k, v in log_vars.items() if 'loss' in k)
    log_vars['loss'] = loss
    return loss, {k: v.item() for k, v in log_vars.items()}


@torch.no_grad()
def validate(model, val_loader, val_dataset, eval_cfg, logger):
    """Run validation and return eval_results dict."""
    model.eval()
    results = []

    for data in tqdm(val_loader, desc='Validation'):
        preds = model(data['keypoint'].cuda(), return_loss=False)
        results.extend(preds)
    
    eval_results = val_dataset.evaluate(
        results,
        metrics=eval_cfg.get('metrics', ['top_k_accuracy', 'mean_class_accuracy']),
        metric_options={'top_k_accuracy': {'topk': eval_cfg.get('topk', (1, 5))}}
    )

    for name, val in eval_results.items():
        logger.info(f'{name}: {val:.4f}')
    
    return eval_results


def save_checkpoint(model, optimizer, epoch, work_dir, filename, **kwargs):
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        **kwargs
    }, osp.join(work_dir, filename))


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Work directory
    work_dir = args.work_dir or cfg.get('work_dir') or osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)

    # Logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger('protogcn')
    logger.addHandler(logging.FileHandler(osp.join(work_dir, f'{timestamp}.log')))

    # Seed
    seed = args.seed if args.seed is not None else np.random.randint(2**31)
    set_random_seed(seed, args.deterministic)
    logger.info(f'Seed: {seed}')

    #==================================Dataset================================
    data_cfg = cfg['data']

    train_dataset = PoseDataset(
        ann_file=data_cfg['train']['ann_file'],
        pipeline=data_cfg['train']['pipeline'],
        split=data_cfg['train'].get('split'),
        data_prefix=data_cfg.get('data_prefix', ''),
        test_mode=False
    )
    val_dataset = PoseDataset(
        ann_file=data_cfg['val']['ann_file'],
        pipeline=data_cfg['val']['pipeline'],
        split=data_cfg['val'].get('split'),
        data_prefix=data_cfg.get('data_prefix', ''),
        test_mode=True
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=data_cfg.get('video_per_gpu', 64),
        num_workers=data_cfg.get('workers_per_gpu', 4),
        shuffle=True, seed=seed, drop_last=True
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=data_cfg.get('test_dataloader', {}).get('video_per_gpu', 1),
        num_workers=data_cfg.get('workers_per_gpu', 4),
        shuffle=False
    )

    logger.info(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}')

    #================================= Model =================================
    model_cfg = cfg['model']
    backbone_cfg = {k: v for k, v in model_cfg['backbone'].items() if k != 'type'}
    head_cfg = {k: v for k, v in model_cfg['cls_head'].items() if k != 'type'}

    model = Recognizer(
        backbone=ProtoGCN(**backbone_cfg),
        cls_head=Head(**head_cfg),
        train_cfg=model_cfg.get('train_cfg'),
        test_cfg=model_cfg.get('test_cfg')
    ).cuda()

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    #========================= Optimizer & Scheduler ========================
    opt_cfg = cfg['optimizer']
    optimizer = optim.SGD(
        model.parameters(),
        lr=opt_cfg.get('lr', 0.1),
        momentum=opt_cfg.get('momentum', 0.9),
        weight_decay=opt_cfg.get('weight_decay', 0.0005),
        nesterov=opt_cfg.get('nesterov', True)
    )

    total_epochs = cfg.get('total_epochs', 150)
    total_iters = total_epochs * len(train_loader)
    min_lr = cfg.get('lr_config', {}).get('min_lr', 0)
    base_lr = opt_cfg['lr']

    #================================ Resume =================================
    start_epoch, best_score, best_epoch, current_iter = 0, 0, 0, 0

    resume_from = args.resume_from
    if resume_from is None and cfg.get('auto_resume', True):
        latest = osp.join(work_dir, 'latest.pth')
        if osp.exists(latest):
            resume_from = latest

    if resume_from:
        ckpt = torch.load(resume_from, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        current_iter = ckpt.get('iter', start_epoch * len(train_loader))
        best_score = ckpt.get('best_score', 0)
        best_epoch = ckpt.get('best_epoch', 0)
        logger.info(f'Resumed from epoch {start_epoch}')
    elif args.load_from:
        model.load_state_dict(torch.load(args.load_from, map_location='cpu')['state_dict'])
        logger(f'Loaded from {args.load_from}')

    #=============================== Training ===============================
    eval_cfg = cfg.get('evaluation', {})
    eval_interval = eval_cfg.get('interval', 1)
    log_interval = cfg.get('log_config', {}).get('interval', 100)
    ckpt_interval = cfg.get('checkpoint_config', {}).get('interval', 1)

    logger.info(f'Start training: {total_epochs} epochs')

    for epoch in range(start_epoch, total_epochs):
        model.train()
        log_vars_sum, num_samples = {}, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')

        for i, data in enumerate(pbar):
            current_iter += 1

            # Cosine Annealing LR (by iter)
            progress = current_iter / total_iters
            lr = min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward & Backward
            optimizer.zero_grad()
            losses = model(data['keypoint'].cuda(), data['label'].cuda(), return_loss=True)
            loss, log_vars = parse_losses(losses)
            loss.backward()
            optimizer.step()

            # Accumulate logs
            bs = data['keypoint'].size(0)
            for k, v in log_vars.items():
                log_vars_sum[k] = log_vars_sum.get(k, 0) + v * bs
            num_samples += bs

            # Log
            if (i+1) % log_interval == 0:
                avg_loss = log_vars_sum.get('loss', 0) / num_samples
                avg_top1 = log_vars_sum.get('top1_loss', 0) / num_samples
                pbar.set_postfix(loss=f'{avg_loss:.4f}', top1=f'{avg_top1:.4f}', lr=f'{lr:.6f}')
                logger.info(f'Epoch [{epoch+1}][{i+1}/{len(train_loader)}] '
                            f'lr: {lr:.6f}, loss: {avg_loss:.4f}, top1: {avg_top1:.4f}')
                log_vars_sum, num_samples = {}, 0
        
        # Validataion
        if args.validate and (epoch+1) % eval_interval == 0:
            eval_results = validate(model, val_loader, val_dataset, eval_cfg, logger)

            if eval_results.get('top1_acc', 0) > best_score:
                best_score = eval_results['top1_acc']
                best_epoch = epoch + 1
                save_checkpoint(model, optimizer, epoch+1, work_dir,
                                f'best_top1_acc_epoch_{epoch+1}.pth',
                                iter=current_iter, best_score=best_score, best_epoch=best_epoch)
                logger.info(f'New best: {best_score:.4f} at epoch {epoch+1}')
        
        # Save checkpoint
        if (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(model, optimizer, epoch+1, work_dir, f'epoch_{epoch+1}.pth',
                            iter=current_iter, best_score=best_score, best_epoch=best_epoch)
            
        save_checkpoint(model, optimizer, epoch+1, work_dir, f'latest.pth',
                        iter=current_iter, best_score=best_score, best_epoch=best_epoch)
        
    logger.info(f'Training done. Best: {best_score:.4f} at epoch {best_epoch}')


    #=============================== Final Test =================================
    if args.test_last or args.test_best:
        eval_cfg = cfg.get('evaluation', {})
        test_cfg = data_cfg.get('test', data_cfg['val'])
        test_dataset = PoseDataset(
            ann_file=test_cfg['ann_file'],
            pipeline=test_cfg['pipeline'],
            split=test_cfg.get('split'),
            data_prefix=data_cfg.get('data_prefix', ''),
            test_mode=True
        )
        test_loader = build_dataloader(
            test_dataset,
            batch_size=data_cfg.get('test_dataloader', {}).get('videos_per_gpu', 1),
            num_workers=data_cfg.get('workers_per_gpu', 4),
            shuffle=False
        )

        if args.test_last and osp.exists(osp.join(work_dir, 'latest.pth')):
            model.load_state_dict(torch.load(osp.join(work_dir, 'latest.pth'))['state_dict'])
            results = validate(model, test_loader, test_dataset, eval_cfg, logger)
            dump_file(results, osp.join(work_dir, 'last_pred.pkl'))
        
        if args.test_best:
            best_ckpts = [f for f in os.listdir(work_dir) if 'best' in f and f.endswith('.pth')]
            if best_ckpts:
                best_ckpt = max(best_ckpts, key=lambda x: int(x.split('epoch_')[-1].replace('.pth', '')) if 'epoch_' in x else 0)
                model.load_state_dict(torch.load(osp.join(work_dir, best_ckpt))['state_dict'])
                results = validate(model, test_loader, test_dataset, eval_cfg, logger)
                dump_file(results, osp.join(work_dir, 'best_pred.pkl'))


if __name__ == '__main__':
    main()
    
