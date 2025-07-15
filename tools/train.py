# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import argparse
import os
import sys
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR
from utils.logger import setup_logger

def fix_initial_lr(optimizer):
    """Fix missing initial_lr in optimizer param_groups."""
    for param_group in optimizer.param_groups:
        if 'initial_lr' not in param_group:
            param_group['initial_lr'] = param_group['lr']
def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)
    model.to(cfg.MODEL.DEVICE)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(cfg.MODEL.DEVICE)

    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        
        optimizer = make_optimizer(cfg, model)
        loss_func, _ = make_loss(cfg, num_classes) 

        start_epoch = 0
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location=cfg.MODEL.DEVICE)
            start_epoch = checkpoint.get('epoch', 0)
            print('Start epoch:', start_epoch)

            # Cargamos solo los pesos del modelo (sin classifier)
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys when loading state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys when loading state_dict: {unexpected_keys}")
            print(f"[Fine-tuning] Ignorando pesos de 'classifier'. Adaptado para {model.classifier.out_features} clases.")

            # 🚫 No cargamos el optimizador, porque el número de clases probablemente ha cambiado

        fix_initial_lr(optimizer)

        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            cfg.SOLVER.WARMUP_FACTOR,
            cfg.SOLVER.WARMUP_ITERS,
            cfg.SOLVER.WARMUP_METHOD,
            start_epoch
        )

        do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_func, num_query, start_epoch)

    elif cfg.MODEL.IF_WITH_CENTER == 'yes':
        print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        loss_func, center_criterion = make_loss_with_center(cfg, num_classes)
        optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)

        start_epoch = 0
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            checkpoint = torch.load(cfg.MODEL.PRETRAIN_PATH)
            start_epoch = checkpoint.get('epoch', 0)
            print('Start epoch:', start_epoch)

            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                print(f"[WARNING] Missing keys when loading state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"[WARNING] Unexpected keys when loading state_dict: {unexpected_keys}")
            print(f"[Fine-tuning] Ignorando pesos de 'classifier'. Adaptado para {model.classifier.out_features} clases.")

            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'center_param' in checkpoint:
                try:
                    center_criterion.load_state_dict(checkpoint['center_param'])
                except Exception as e:
                    print(f"[WARNING] No se pudo cargar center_param completo: {e}")
                    print(f"[Fine-tuning] Inicializando center_param de cero para nuevas clases.")
                    optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
            else:
                print("[WARNING] No center_param en checkpoint, inicializando.")

            if 'optimizer_center' in checkpoint:
                try:
                    optimizer_center.load_state_dict(checkpoint['optimizer_center'])
                except Exception as e:
                    print(f"[WARNING] No se pudo cargar optimizer_center: {e}. Recreando optimizer_center.")
                    optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)

        fix_initial_lr(optimizer)  # <-- TAMBIÉN AQUÍ

        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS,
                                      cfg.SOLVER.WARMUP_METHOD, start_epoch)

        do_train_with_center(cfg, model, center_criterion, train_loader, val_loader,
                             optimizer, optimizer_center, scheduler, loss_func, num_query, start_epoch)

    else:
        raise ValueError("Unsupported value for cfg.MODEL.IF_WITH_CENTER: {}".format(cfg.MODEL.IF_WITH_CENTER))

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ.get("WORLD_SIZE", 1))

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train(cfg)

if __name__ == '__main__':
    main()

