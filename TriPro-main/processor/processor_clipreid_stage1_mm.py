import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from tqdm import tqdm


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank,
                    label2pid,
                    pid2attrs,
                    image_pth="./image_features_stage1.pth"):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    loss_id_meter = AverageMeter()   # 新增：用于专门记录 ID Loss
    loss_attr_meter = AverageMeter() # 新增：用于专门记录 Attribute Loss

    scaler = torch.cuda.amp.GradScaler()
    xent = SupConLoss(device)

    import os
    import time
    from datetime import timedelta
    save_path = os.path.join(image_pth)
    all_start_time = time.monotonic()
    
    # ------------------- 特征提取与加载阶段 -------------------
    if os.path.exists(save_path):
        logger.info(f"Loading precomputed features from {save_path}")
        features = torch.load(save_path)
        labels_list = features['labels'].cuda()  # Tensor [N]
        image_features_list_rgb = features['rgb'].cuda()  # Tensor [N, D]
        image_features_list_eve = features['eve'].cuda()  # Tensor [N, D]
        image_pids = features['pids']  # List[str]
    else:
        logger.info("Start Stage 1 feature extraction...")
        image_features_rgb = []
        image_features_eve = []
        labels = []
        image_pids = []

        with torch.no_grad():
            for (img, vid, target_cam, target_view) in tqdm(train_loader_stage1, desc="Stage 1 Image Feature Extract"):
                pid_str = str(label2pid[int(vid)]).zfill(4)
                image_pids.append(pid_str)

                img = img.to(device)
                target = vid.to(device)
                if len(img.size()) - 1 == 6:
                    b, m, n, t, c, h, w = img.size()
                    assert (b == 1)
                    img = img.view(b, m, n * t, c, h, w)
                with torch.amp.autocast('cuda'):
                    image_feature_rgb, image_feature_eve = model(x=img, label=target, get_image=True)
                    for i, img_feat_rgb, img_feat_eve in zip(target, image_feature_rgb, image_feature_eve):
                        labels.append(i)
                        image_features_rgb.append(img_feat_rgb.cpu())
                        image_features_eve.append(img_feat_eve.cpu())
                        
        labels_list = torch.stack(labels, dim=0).cuda()
        image_features_list_rgb = torch.stack(image_features_rgb, dim=0).cuda()
        image_features_list_eve = torch.stack(image_features_eve, dim=0).cuda()
        del labels, image_feature_rgb, image_feature_eve
        
        # 保存特征缓存
        torch.save({
            'labels': labels_list.cpu(),
            'rgb': image_features_list_rgb.cpu(),
            'eve': image_features_list_eve.cpu(),
            'pids': image_pids
        }, save_path)
        logger.info(f"Saved features to {save_path}")

    # ------------------- 模型主训练循环 (ID + Attribute 联合优化) -------------------
    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
    num_image = labels_list.shape[0]
    i_ter = num_image // batch
    
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_id_meter.reset()
        loss_attr_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list]
            target_attrs = torch.stack([pid2attrs[image_pids[b]] for b in b_list]).cuda()
            image_features_rgb = image_features_list_rgb[b_list]
            image_features_eve = image_features_list_eve[b_list]
            
            # 核心修改点 1: 一次性提取基础文本特征 (text_features) 和属性文本特征 (attrs_feat)
            with torch.amp.autocast('cuda'):
                text_features, attrs_feat = model(label=target, target_attrs=target_attrs, get_text=True)
                # 对属性特征求均值，将维度与图像特征对齐
                attrs_feat = attrs_feat.mean(dim=1) 

            # 核心修改点 2: 计算基础的身份对比损失 (ID Loss)
            loss_i2t_rgb = xent(image_features_rgb, text_features, target, target)
            loss_t2i_rgb = xent(text_features, image_features_rgb, target, target)
            loss_i2t_eve = xent(image_features_eve, text_features, target, target)
            loss_t2i_eve = xent(text_features, image_features_eve, target, target)
            loss_id = (loss_i2t_rgb + loss_t2i_rgb + loss_i2t_eve + loss_t2i_eve) / 2

            # 核心修改点 3: 激活被封印的属性对比损失 (Attribute Loss)
            loss_i2t_rgb_attr = xent(image_features_rgb, attrs_feat, target, target)
            loss_t2i_rgb_attr = xent(attrs_feat, image_features_rgb, target, target)
            loss_i2t_eve_attr = xent(image_features_eve, attrs_feat, target, target)
            loss_t2i_eve_attr = xent(attrs_feat, image_features_eve, target, target)
            loss_attr = (loss_i2t_rgb_attr + loss_t2i_rgb_attr + loss_i2t_eve_attr + loss_t2i_eve_attr) / 2

            # 核心修改点 4: 损失联合，给属性 Loss 加了 0.5 的权重防止训练初期梯度爆炸
            loss = loss_id + 0.5 * loss_attr

            # 反向传播与优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录各个 Loss 的平均值
            loss_meter.update(loss.item(), batch)
            loss_id_meter.update(loss_id.item(), batch)
            loss_attr_meter.update(loss_attr.item(), batch)
            
            torch.cuda.synchronize()
            
            # 日志打印：现在可以清晰地看到 Total Loss, ID Loss 和 Attr Loss 的下降情况
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Total Loss: {:.3f} (ID:{:.3f}, Attr:{:.3f}), Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, loss_id_meter.avg, loss_attr_meter.avg, scheduler._get_lr(epoch)[0]))

        # 保存权重模型
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
