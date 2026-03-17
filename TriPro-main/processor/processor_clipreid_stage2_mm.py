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
    logger.info('start training')
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
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = torch.cuda.amp.GradScaler()
    xent = SupConLoss(device)

    # train
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

    for epoch in range(1, 120 + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter_rgb.reset()
        acc_meter_eve.reset()
        evaluator.reset()

        model.train()
        text_labels = torch.arange(num_classes).to(device)

        # --- Stage 1 类似的训练过程 ---
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            mid_optimizer.zero_grad()
            img = img.to(device)

            if isinstance(vid, list):
                target = torch.tensor(vid).to(device)
            else:
                target = vid.to(device)

            with amp.autocast(enabled=True):
                image_features_rgb, image_features_eve = model(x=img, label=target, get_image=True)

                loss_i2t_rgb = xent(image_features_rgb, text_features, target, text_labels)
                loss_i2t_eve = xent(image_features_eve, text_features, target, text_labels)
                loss = (loss_i2t_rgb + loss_i2t_eve) / 2
                scaler.scale(loss).backward()
                scaler.step(mid_optimizer)
                scaler.update()
                loss_meter.update(loss.item(), img.shape[0])
                torch.cuda.synchronize()
                if (n_iter + 1) % 450 == 0:
                    logger.info("Prompt-Align Epoch[{}] Iteration[{}/{}] Loss: {:.3f}"
                                .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                        loss_meter.avg))

        if epoch % 120 == 0 or epoch % 60 == 0 or epoch == 1:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_stage2.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}_stage2.pth'.format(epoch)))

    # --- 正式训练循环 ---
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter_rgb.reset()
        acc_meter_eve.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
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

            # [关键] 训练时不一定需要 bool，但为了保险起见，或者如果训练也崩了可以加 .bool()
            # 根据报错，训练没崩，暂时保持原样或统一加 bool。这里先保持原样，因为数据源不同
            target_attrs = torch.stack(attr_list).cuda()

            optimizer.zero_grad()
            optimizer_center.zero_grad()
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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
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

                # [核心修复] 这里添加了 .bool()
                # 解释: 报错说 "expected boolean tensor, got Long"，所以这里必须强转
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

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
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

        # [核心修复] 这里也添加了 .bool()
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