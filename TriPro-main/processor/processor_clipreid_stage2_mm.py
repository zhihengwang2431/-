import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import json
import random


def do_train_stage2(cfg,
                    model,
                    center_criterion,
                    train_loader_stage2,
                    val_loader,
                    mid_optimizer,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query, local_rank,
                    label2pid,
                    pid2attrs,
                    test_attrs):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start Stage 2 training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter_rgb = AverageMeter()
    acc_meter_eve = AverageMeter()

    # 【修复 1：Re-ranking 入口开启】将 YAML 中的 RE_RANKING 开关传入评估器
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    scaler = torch.cuda.amp.GradScaler()
    xent = SupConLoss(device)

    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # 预计算文本特征
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    # 【神级优化：已彻底删除原作者长达 120 轮的冗余 Prompt-Align 循环，防止一上来就 OOM】

    # --- 正式训练循环 ---
    # 【修复 2：自适应梯度累加】目标有效 Batch Size 设为 64
    target_effective_batch = 64
    actual_batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    accumulation_steps = max(1, target_effective_batch // actual_batch)
    logger.info(
        f"==> Target Batch: {target_effective_batch}, GPU Batch: {actual_batch}, Accumulation Steps: {accumulation_steps}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter_rgb.reset()
        acc_meter_eve.reset()
        evaluator.reset()

        scheduler.step()
        model.train()

        # 【核心：确保在每个 Epoch 开始时，梯度是干净的】
        optimizer.zero_grad()
        if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            optimizer_center.zero_grad()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):

            if isinstance(vid, list):
                current_batch_labels = vid
                target = torch.tensor(vid).to(device)
            else:
                current_batch_labels = vid.tolist()
                target = vid.to(device)

            attr_list = []
            default_attr_train = torch.zeros_like(list(pid2attrs.values())[0])

            for label in current_batch_labels:
                found = False
                if label in label2pid:
                    real_pid = label2pid[label]
                    keys_to_try = [str(real_pid).zfill(4), str(real_pid)]
                    for k in keys_to_try:
                        if k in pid2attrs:
                            attr_list.append(pid2attrs[k])
                            found = True
                            break

                if not found:
                    attr_list.append(default_attr_train)

            target_attrs = torch.stack(attr_list).cuda()

            img = img.to(device)

            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else:
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None

            with amp.autocast(enabled=True):
                score, feat, image_features = model(x=img, label=target, cam_label=target_cam, view_label=target_view,
                                                    target_attrs=target_attrs)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

                # 【核心：Loss 必须除以累加步数，防止梯度成倍爆炸】
                loss = loss / accumulation_steps

            # 反向传播算梯度（只计算，不更新，梯度会暂时在显存中叠加）
            scaler.scale(loss).backward()

            # 【核心：当攒够了 accumulation_steps 步，或者到了该 Epoch 最后一个 Batch，才统一更新权重】
            if ((n_iter + 1) % accumulation_steps == 0) or ((n_iter + 1) == len(train_loader_stage2)):
                scaler.step(optimizer)
                scaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()

                # 更新完毕后，立刻清空暂存的梯度，准备下一轮“积攒”
                optimizer.zero_grad()
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    optimizer_center.zero_grad()

            acc = (logits.max(1)[1] == target).float().mean()
            # 记录日志时，把 loss 乘回来以显示正常的数值大小
            loss_meter.update(loss.item() * accumulation_steps, img.shape[0])
            acc_meter_rgb.update(acc, 1)
            torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter_rgb.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        # --- 验证循环 ---
        if epoch % eval_period == 0:
            if not os.path.exists(test_attrs):
                raise FileNotFoundError("Test attribute file not found!")

            with open(test_attrs, 'r') as f:
                all_test_attrs = json.load(f)

            first_key = list(all_test_attrs.keys())[0]
            val_default_attr = torch.zeros_like(torch.tensor(all_test_attrs[first_key]))

            model.eval()
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):

                if isinstance(vid, list):
                    current_batch_pids = vid
                else:
                    current_batch_pids = vid.tolist()

                val_attr_list = []
                for pid in current_batch_pids:
                    found = False
                    keys_to_try = [str(pid).zfill(4), str(pid)]
                    for k in keys_to_try:
                        if k in all_test_attrs:
                            val_attr_list.append(torch.tensor(all_test_attrs[k]))
                            found = True
                            break
                    if not found:
                        val_attr_list.append(val_default_attr)

                # 保持了之前解决数据类型不匹配的 .bool() 修复
                target_attrs = torch.stack(val_attr_list).cuda().bool()

                if target_attrs.size(0) > img.size(0):
                    target_attrs = target_attrs[:img.size(0)]

                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else:
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else:
                        target_view = None
                    feat = model(img, cam_label=camids, view_label=target_view, target_attrs=target_attrs)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference(cfg,
                 model,
                 test_attrs,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    with open(test_attrs, 'r') as f:
        all_test_attrs = json.load(f)

    first_key = list(all_test_attrs.keys())[0]
    val_default_attr = torch.zeros_like(torch.tensor(all_test_attrs[first_key]))

    # 【修复 3：推理阶段也要开启 Re-ranking】
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        if isinstance(vid, list):
            current_batch_pids = vid
        else:
            current_batch_pids = vid.tolist()

        val_attr_list = []
        for pid in current_batch_pids:
            found = False
            keys_to_try = [str(pid).zfill(4), str(pid)]
            for k in keys_to_try:
                if k in all_test_attrs:
                    val_attr_list.append(torch.tensor(all_test_attrs[k]))
                    found = True
                    break
            if not found:
                val_attr_list.append(val_default_attr)

        target_attrs = torch.stack(val_attr_list).cuda().bool()

        if target_attrs.size(0) > img.size(0):
            target_attrs = target_attrs[:img.size(0)]

        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else:
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else:
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view, target_attrs=target_attrs)
            evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
