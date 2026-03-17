import os
import argparse
import torch
import torch.nn as nn
from config import cfg_mars
from datasets.make_dataloader import make_dataloader
from model.make_model_clipreid_mm import make_model_mm
from utils.logger import setup_logger
from processor.processor_clipreid_stage2_mm import do_inference
import os.path as osp

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg_mars.merge_from_file(args.config_file)
    cfg_mars.merge_from_list(args.opts)
    
    # 强制开启评估模式
    cfg_mars.defrost()
    cfg_mars.TEST.EVAL = True
    cfg_mars.freeze()

    output_dir = cfg_mars.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))

    # 1. 构建数据加载器 (只关心 val_loader 和 query/gallery 信息)
    # 注意：make_dataloader 返回多个值，我们需要按顺序接收
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg_mars)

    # 2. 构建模型
    model = make_model_mm(cfg_mars, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 3. 加载权重
    # 优先使用命令行指定的 weight，如果没有则使用 config 里的
    if cfg_mars.TEST.WEIGHT == '':
        logger.warning("No weight path specified in config!")
    
    weight_path = cfg_mars.TEST.WEIGHT
    if osp.exists(weight_path):
        logger.info(f"Loading weights from: {weight_path}")
        model.load_param(weight_path)
    else:
        logger.error(f"Weight file not found: {weight_path}")
        return

    # 4. 执行推理
    # 假设 processor_clipreid_stage2_mm.py 中包含 do_inference 函数
    # 这是绝大多数 ReID 代码库的惯例
    try:
        do_inference(cfg_mars, model, val_loader, num_query)
    except NameError:
        # 如果 processor 里没有 do_inference，可能需要手动写 inference loop，
        # 但通常 stage2 的 processor 里会有 validation 的逻辑。
        logger.error("Could not find do_inference function. Please check processor code.")

if __name__ == '__main__':
    main()