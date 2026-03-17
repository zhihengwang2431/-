from utils.logger import setup_logger
from datasets.make_dataloader import make_dataloader
from model.make_model_clipreid_mm import make_model_mm
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage, make_optimizer_mid_stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1_mm import do_train_stage1
from processor.processor_clipreid_stage2_mm import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg_mars
import os.path as osp
from scipy.io import loadmat
import json

import ast

def bool_str_to_tensor(data) -> torch.BoolTensor:
    # 既然 json.load 已经读成了 list，直接转 tensor 即可
    if isinstance(data, list):
        return torch.tensor(data, dtype=torch.bool)
    
    # 只有当它是字符串的时候，才进行 strip 和 eval
    if isinstance(data, str):
        bool_list = ast.literal_eval(data.strip())
        return torch.tensor(bool_list, dtype=torch.bool)
        
    raise TypeError(f"Unsupported type for attribute data: {type(data)}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    # 修改点 1: 去掉这里硬编码的长路径，或者你可以改成你自己的默认路径，
    # 但最安全的是留空，或者依赖命令行参数
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg_mars.merge_from_file(args.config_file)
    cfg_mars.merge_from_list(args.opts)
    cfg_mars.freeze()

    set_seed(cfg_mars.SOLVER.SEED)

    if cfg_mars.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg_mars.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg_mars.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg_mars))

    if cfg_mars.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg_mars)

    model = make_model_mm(cfg_mars, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg_mars, num_classes=num_classes)

    optimizer_1stage = make_optimizer_1stage(cfg_mars, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg_mars.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg_mars.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg_mars.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg_mars.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)
    
    # --- 修改点 2: 动态拼接 track_train_info_path ---
    # 原始的硬编码路径: track_train_info_path = osp.join('/media/amax/...', 'info/tracks_train_info.mat')
    # 修改为使用 cfg_mars.DATASETS.ROOT_DIR
    # 注意: ROOT_DIR 在你的配置里是 '/data1/zjj_data'，所以需要拼接 'mars' 目录
    # 如果你的 ROOT_DIR 已经是 '/data1/zjj_data/mars'，则去掉下面的 'mars' 这一层
    data_root = cfg_mars.DATASETS.ROOT_DIR
    mars_root = osp.join(data_root, 'mars')
    track_train_info_path = osp.join(mars_root, 'info/tracks_train_info.mat')
    
    # 打印一下路径，方便出错时排查
    logger.info(f"Loading track info from: {track_train_info_path}")
    # -----------------------------------------------

    track_train = loadmat(track_train_info_path)['track_train_info']
    train_pid_list = list(set(track_train[:, 2].tolist()))
    label2pid = {label: pid for label, pid in enumerate(train_pid_list)}
    # --- 修改开始 ---
    with open(cfg_mars.DATASETS.TRAIN_ATTR, 'r', encoding='utf-8') as f:
        train_pid2attrs = json.load(f)
    pid2attrs = {}
    
    for k, v in train_pid2attrs.items():
        # 问题核心：JSON Key 是 '0001C1T0001' (Tracklet ID)
        # 我们需要提取前 4 位 '0001' (Person ID) 作为 Key
        
        k_str = str(k).strip()
        
        # 1. 提取 PID: 如果 key 长度足够长（比如 Tracklet 格式），取前4位
        if len(k_str) >= 4:
            new_k = k_str[:4]
        else:
            # 兼容万一已经是 PID 的情况
            new_k = k_str.zfill(4)

        # 2. 如果这个 PID 已经在字典里了，跳过 (因为同一个人的属性是一样的，存一份就行)
        if new_k in pid2attrs:
            continue
            
        # 3. 转换 Value 为 Tensor
        if isinstance(v, list):
            pid2attrs[new_k] = torch.tensor(v, dtype=torch.bool)
        else:
            import ast
            pid2attrs[new_k] = torch.tensor(ast.literal_eval(v), dtype=torch.bool)
            
    # 再次打印调试，这次你应该能看到 Keys 变成了 ['0001', '0002', ...]
    logger.info(f"DEBUG Processed Keys (first 5): {list(pid2attrs.keys())[:5]}")
    # --- 修改结束 ---

        
    do_train_stage1(
        cfg_mars,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank,
        label2pid,
        pid2attrs,
        image_pth=cfg_mars.DATASETS.STAGE1_IMG_FEAT
    )
    mid_optimizer = make_optimizer_mid_stage(cfg_mars, model)

    
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg_mars, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg_mars.SOLVER.STAGE2.STEPS, cfg_mars.SOLVER.STAGE2.GAMMA, cfg_mars.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg_mars.SOLVER.STAGE2.WARMUP_ITERS, cfg_mars.SOLVER.STAGE2.WARMUP_METHOD)
    do_train_stage2(
        cfg_mars,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        mid_optimizer,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank, label2pid, pid2attrs, cfg_mars.DATASETS.TEST_ATTR
    )