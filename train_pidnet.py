import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
from functions_ori_edge import *
from pathlib import Path
from tensorboardX import SummaryWriter
import math
import models
from utils.utils import FullModel
from collections import OrderedDict

from kd_losses.DCSFKDLoss import DCSFKDLoss
from kd_losses.OutputKDLoss import OutputKDLoss


def _update_confmat(confmat, preds, targets, num_classes, ignore_index=255):
    valid = (targets != ignore_index)
    if not valid.any():
        return confmat
    t = targets[valid].view(-1)
    p = preds[valid].view(-1)
    k = t * num_classes + p
    hist = torch.bincount(k, minlength=num_classes * num_classes)
    hist = hist.view(num_classes, num_classes).to(confmat.device)
    confmat += hist.to(dtype=confmat.dtype)
    return confmat


def compute_miou_from_confmat(confmat):
    confmat = confmat.to(torch.float64)
    TP = torch.diag(confmat)
    FP = confmat.sum(0) - TP
    FN = confmat.sum(1) - TP
    denom = TP + FP + FN
    ious = torch.where(denom > 0, TP / denom.clamp(min=1), torch.full_like(TP, float('nan')))
    miou = torch.nanmean(ious)
    iou_list = [float(v) if not torch.isnan(v) else float('nan') for v in ious]
    return float(miou), iou_list


def compute_pixel_accuracy_from_confmat(confmat):
    total = confmat.sum().clamp(min=1)
    correct = torch.trace(confmat)
    return float((correct / total).item())


def _seed_worker(worker_id):
    import random, numpy as np, torch
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def train(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    AUG_NAMES = ["haze", "rain", "raindrop", "low_light", "overbright"]
    name_to_idx = {n: i for i, n in enumerate(AUG_NAMES)}

    # -------------------- Dataset & Dataloader --------------------
    train_dataset = SegmentationDataset(
        args.dataset_dir, args.crop_size, 'train', args.scale_range,
        val_resize_size=(1080, 1920),
        normal_aug_prob=args.normal_aug_prob,
        severity_range=(args.severity_min, args.severity_max),
    )
    display_dataset_info(args.dataset_dir, train_dataset)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank,
                                       drop_last=True, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=8, pin_memory=True, worker_init_fn=_seed_worker, collate_fn=collate_with_meta)

    val_dataset = SegmentationDataset(args.dataset_dir, args.crop_size, 'val', args.scale_range,
                                      val_resize_size=(1080, 1920))
    display_dataset_info(args.dataset_dir, val_dataset)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank,
                                     drop_last=False, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size//2), sampler=val_sampler,
                            num_workers=8, pin_memory=True, worker_init_fn=_seed_worker, collate_fn=collate_with_meta)
    
    # loss
    criterion = CrossEntropy(ignore_label=255)
    bd_criterion = BondaryLoss()

    # kd loss
    criterion_kd = DCSFKDLoss(loss_weight=1.0).to(device)
    criterion_output_kd = OutputKDLoss(temperature=4.0, loss_weight=0.1, ignore_index=255).to(device)
    
    # Model
    print(f"[GPU {local_rank }] Before model setup")
    model_base = models.pidnet.get_seg_model(num_classes=args.num_classes, load_path=args.loadpath, name='pidnet_s')
    model = FullModel(model_base, criterion, bd_criterion).to(device)
    model = DDP(model, device_ids=[local_rank])
    print(f"[GPU {local_rank }] DDP initialized")

    # Teacher
    teacher_model_base = models.pidnet.get_seg_model(num_classes=args.num_classes, load_path=args.teacher_loadpath, name='pidnet_l')
    teacher_model = FullModel(teacher_model_base, criterion, bd_criterion, is_teacher=True).to(device)
    teacher_model = DDP(teacher_model, device_ids=[local_rank])

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # Optimizer, Scheduler
    param_groups = [{'params': model.parameters()}]
    if any(p.requires_grad for p in criterion_kd.parameters()):
        param_groups.append({'params': criterion_kd.parameters()})
    if any(p.requires_grad for p in getattr(criterion_output_kd, 'parameters', lambda: [])()):
        if any(p.requires_grad for p in criterion_output_kd.parameters()):
            param_groups.append({'params': criterion_output_kd.parameters()})

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-3)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs=args.epochs, warmup_epochs=10, eta_min=1e-5)
    # scheduler = WarmupPolyEpochLR(optimizer, total_epochs=args.epochs, warmup_epochs=5, warmup_ratio=5e-4)

    # -------------------- Logging/TensorBoard --------------------
    writer = None
    os.makedirs(args.result_dir, exist_ok=True)
    log_path = os.path.join(args.result_dir, "log.txt")
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("Epoch\tTrain-loss\tVal-loss\tmIoU\tAcc\tlearningRate\n")
    if local_rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.result_dir, "tb"))

    def _get_state_dict(m):
        return m.module.state_dict() if isinstance(m, DDP) else m.state_dict()

    # ---------- NEW: prefix 정규화 유틸 ----------
    def _strip_prefixes(sd, prefixes):
        new_sd = OrderedDict()
        for k, v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[nk] = v
        return new_sd

    def _best_load_into_base(base, state):
        """
        base: PIDNet base module (FullModel 내부 .model)
        state: checkpoint state_dict (as-is)
        여러 전략을 시도해 '누락 키 수'가 가장 적은 로드를 채택.
        반환: (missing, unexpected)
        """
        strategies = []
        # 0) as-is
        strategies.append(("as-is", state))
        # 1) strip module.model. → module. → model.
        strategies.append(("strip-mm", _strip_prefixes(state, ["module.model.", "module.", "model."])))
        # 2) strip module. → model.
        strategies.append(("strip-m", _strip_prefixes(state, ["module.", "model."])))
        # 3) strip model. only
        strategies.append(("strip-model", _strip_prefixes(state, ["model."])))

        best = None
        best_missing = None
        best_unexpected = None
        best_name = None

        for name, sd_try in strategies:
            try:
                missing, unexpected = base.load_state_dict(sd_try, strict=False)
                miss_n = len(missing)
                # pick best (fewest missing)
                if best is None or miss_n < len(best_missing):
                    best = sd_try
                    best_missing = missing
                    best_unexpected = unexpected
                    best_name = name
                # revert to avoid side effects before final load
                base.load_state_dict(base.state_dict())
            except Exception:
                continue

        # 최종 적용
        if best is not None:
            missing, unexpected = base.load_state_dict(best, strict=False)
            if local_rank == 0:
                print(f"[Resume] load strategy: {best_name} | missing={len(missing)}, unexpected={len(unexpected)}")
            return missing, unexpected
        # fallback: as-is 적용 시도
        missing, unexpected = base.load_state_dict(state, strict=False)
        if local_rank == 0:
            print(f"[Resume] load strategy: fallback-as-is | missing={len(missing)}, unexpected={len(unexpected)}")
        return missing, unexpected

    def _load_model_state(m, state):
        """
        m: DDP(FullModel)
        state: weights-only dict or nested dict['model']
        FullModel 내부의 베이스 모델은 m.module.model
        """
        target = m.module if isinstance(m, DDP) else m  # FullModel
        base = getattr(target, "model", None)           # PIDNet base
        if base is None:
            # 베이스가 없으면 전체에 관용 로드
            missing, unexpected = target.load_state_dict(state, strict=False)
            if local_rank == 0:
                print(f"[Resume] loaded into FullModel | missing={len(missing)}, unexpected={len(unexpected)}")
            return

        # 베이스에 최적 전략으로 로드
        missing, unexpected = _best_load_into_base(base, state)
        if local_rank == 0:
            if missing:
                print(f"[Resume] Missing keys: {len(missing)} (showing 10) -> {missing[:10]}")
            if unexpected:
                print(f"[Resume] Unexpected keys: {len(unexpected)} (showing 10) -> {unexpected[:10]}")

    def _is_full_checkpoint(obj):
        return isinstance(obj, dict) and "model" in obj

    # -------------------- Resume / Load --------------------
    start_epoch = 0
    best_miou = float("-inf")

    kd_state_pending = None
    kd_out_state_pending = None

    def _try_read_epoch_from_last(result_dir):
        last_path = os.path.join(result_dir, "last.pth.tar")
        if os.path.isfile(last_path):
            obj = torch.load(last_path, map_location={f'cuda:{0}': f'cuda:{local_rank}'})
            if isinstance(obj, dict) and "epoch" in obj:
                return int(obj["epoch"])  # 저장 시 epoch+1(다음 epoch, 0-index)
        return None

    def _try_read_epoch_from_log(path):
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) <= 1:
                return None
            last_line = lines[-1]
            ep_str = last_line.split()[0]   # 1-indexed
            return int(ep_str)              # 다음 epoch(0-index)로 그대로 사용
        except Exception:
            return None

    if args.resume is not None and os.path.isfile(args.resume):
        map_location = {f'cuda:{0}': f'cuda:{local_rank}'}
        ckpt = torch.load(args.resume, map_location=map_location)
        if _is_full_checkpoint(ckpt):
            _load_model_state(model, ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0))
            best_miou  = float(ckpt.get("best_miou", float("-inf")))
            if local_rank == 0:
                print(f"[Resume: full] {args.resume} (next_epoch_idx={start_epoch}, best_mIoU={best_miou:.4f})")
            if "criterion_kd" in ckpt:
                kd_state_pending = ckpt["criterion_kd"]
            if "criterion_output_kd" in ckpt:
                kd_out_state_pending = ckpt["criterion_output_kd"]
        else:
            # weights-only (.pth)
            _load_model_state(model, ckpt)
            if local_rank == 0:
                print(f"[Resume: weights-only] {args.resume} → model weights loaded")

            # 1) 같은 폴더의 last.pth.tar에서 opt/sched/epoch 복구 시도
            last_path = os.path.join(args.result_dir, "last.pth.tar")
            if os.path.isfile(last_path):
                last_obj = torch.load(last_path, map_location=map_location)
                if isinstance(last_obj, dict):
                    if "optimizer" in last_obj:
                        optimizer.load_state_dict(last_obj["optimizer"])
                    if "scheduler" in last_obj:
                        scheduler.load_state_dict(last_obj["scheduler"])
                    if "epoch" in last_obj:
                        start_epoch = int(last_obj["epoch"])
                    best_miou = float(last_obj.get("best_miou", float("-inf")))
                    kd_state_pending = last_obj.get("criterion_kd", None)
                    kd_out_state_pending = last_obj.get("criterion_output_kd", None)
                    if local_rank == 0:
                        print(f"[Resume: pulled opt/sched] from {last_path} (next_epoch_idx={start_epoch})")

            # 2) last가 없으면 log.txt → --resume_epoch 순으로 추론
            if start_epoch == 0:
                guessed = _try_read_epoch_from_log(log_path)
                if guessed is None and args.resume_epoch is not None:
                    guessed = int(args.resume_epoch)  # 1-indexed
                if guessed is not None:
                    start_epoch = int(guessed)
                    if local_rank == 0:
                        print(f"[Resume: inferred] start_epoch set to {start_epoch} from history")

            # 스케줄러 정렬: 사전 step() 없이 last_epoch만 맞춤(경고 제거)
            scheduler.last_epoch = start_epoch - 1
    else:
        # loadpath 초기화는 get_seg_model에서 이미 처리됨
        pass

    # ★ KD warmup: 보관해둔 kd_state가 있을 때만 실행
    if kd_state_pending is not None:
        if local_rank == 0:
            print("[KD Warmup] Taking one batch to initialize alignment layers...")

        # 첫 배치 하나만 가져와서 feature 산출 (train_sampler의 epoch 고정 영향 없음)
        with torch.no_grad():
            # 주의: next(iter(loader))는 이후 for-루프와 별개로 '새로운' 이터레이터이므로
            # 여기서 한 번 꺼내도 본 학습 루프 시작에는 영향 없음.
            try:
                imgs, labels, metas, edges = next(iter(train_loader))
            except StopIteration:
                # 드물게 빈 로더면 val_loader에서라도 한 배치
                imgs, labels, metas, edges = next(iter(val_loader))

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bd_gts = edges.to(device, non_blocking=True)

            # student/teacher 한 번 forward만 (loss/opt 갱신 없음)
            # FullModel 시그니처: student → (loss, outputs, features), teacher → (outputs, features)
            _loss_dummy, out_s, feat_s = model(imgs, labels, bd_gts)
            out_t, feat_t = teacher_model(imgs, labels, bd_gts)

            # PIDNet에서는 메인 세그 feature가 [1] 인덱스
            fs = feat_s[1] if isinstance(feat_s, (list, tuple)) else feat_s
            ft = feat_t[1] if isinstance(feat_t, (list, tuple)) else feat_t

            # warmup으로 align 레이어 생성
            criterion_kd.warmup_from_feats(fs, ft)

        # 모든 rank에서 align 생성 완료 동기화
        dist.barrier()

        # 이제야 KD state 복원 (strict=False로 누락 키 허용)
        try:
            criterion_kd.load_state_dict(kd_state_pending, strict=False)
            if local_rank == 0:
                print("[KD Warmup] criterion_kd state restored after alignment init.")
        except Exception as e:
            if local_rank == 0:
                print(f"[KD Warmup] criterion_kd load skipped: {e}")

        # (옵션) Output KD에 학습 파라미터가 있고 state가 보관되어 있으면 로드
        if kd_out_state_pending is not None:
            try:
                criterion_output_kd.load_state_dict(kd_out_state_pending, strict=False)
                if local_rank == 0:
                    print("[KD Warmup] criterion_output_kd state restored.")
            except Exception as e:
                if local_rank == 0:
                    print(f"[KD Warmup] criterion_output_kd load skipped: {e}")

        # 더 이상 보관 불필요
        kd_state_pending = None
        kd_out_state_pending = None

    eps = 1e-6

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion_kd.train()

        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        num_steps = 0

        train_kd_feat_sum = 0.0
        train_kd_out_sum = 0.0

        aug_counts_local = torch.zeros(len(AUG_NAMES), device=device, dtype=torch.long)

        if local_rank == 0:
            loop = tqdm(train_loader, desc=f"[GPU {local_rank}] Epoch [{epoch + 1}/{args.epochs}]", ncols=110)
        else:
            loop = train_loader

        for i, (imgs, labels, metas, edges) in enumerate(loop):
            optimizer.zero_grad(set_to_none=True)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            bd_gts = edges.to(device, non_blocking=True)
            
            loss, outputs, features_s = model(imgs, labels, bd_gts)
            with torch.no_grad():
                outputs_t, features_t = teacher_model(imgs, labels, bd_gts)

            loss_kd_feat = criterion_kd(features_s, features_t)
            loss_kd_output = criterion_output_kd(outputs[-1], outputs_t[-1], labels=labels)

            train_kd_feat_sum += float(loss_kd_feat.item())
            train_kd_out_sum += float(loss_kd_output.item())

            loss += loss_kd_feat + loss_kd_output
            loss.backward()
            optimizer.step()

            # 증강 카운트
            if isinstance(metas, (list, tuple)):
                for m in metas:
                    for (name, sev) in m.get("applied", []):
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1
            elif isinstance(metas, dict):
                for applied in metas.get("applied", []):
                    for (name, sev) in applied:
                        idx = name_to_idx.get(name)
                        if idx is not None:
                            aug_counts_local[idx] += 1

            total_loss += loss.item()
            num_steps += 1

            if local_rank == 0:
                loop.set_postfix(loss=loss.item(),
                                 loss_kd_f=loss_kd_feat.item(),
                                 loss_kd_o=loss_kd_output.item(),
                                 avg_loss=total_loss / max(1, num_steps),
                                 lr=scheduler.get_last_lr()[0])

        torch.cuda.empty_cache()
        dist.barrier()
        scheduler.step()

        # ------ Train epoch 평균(DDP 전체 평균) ------
        train_loss_sum = torch.tensor([total_loss], device=device)
        train_step_sum = torch.tensor([num_steps], device=device, dtype=torch.float32)

        train_kd_feat_t = torch.tensor([train_kd_feat_sum], device=device, dtype=torch.float32)
        train_kd_out_t = torch.tensor([train_kd_out_sum], device=device, dtype=torch.float32)

        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_step_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_kd_feat_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_kd_out_t, op=dist.ReduceOp.SUM)

        train_loss_epoch = (train_loss_sum / train_step_sum).item()
        train_kd_feat_epoch = (train_kd_feat_t / train_step_sum).item()
        train_kd_out_epoch = (train_kd_out_t / train_step_sum).item()

        # ===== Validation =====
        model.eval()
        criterion_kd.eval()

        val_loss_sum = torch.tensor([0.0], device=device)
        val_batches = torch.tensor([0.0], device=device)
        confmat = torch.zeros((args.num_classes, args.num_classes), device=device, dtype=torch.int64)

        val_kd_feat_sum = torch.tensor([0.0], device=device)
        val_kd_out_sum = torch.tensor([0.0], device=device)

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"[GPU {local_rank}] Validate",
                            ncols=110) if local_rank == 0 else val_loader
            for imgs, labels, metas, edges in val_iter:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                bd_gts = edges.to(device, non_blocking=True)
                
                vloss, logits, v_feature_s = model(imgs, labels, bd_gts)
                v_outputs_t, v_feature_t = teacher_model(imgs, labels, bd_gts)

                vloss_kd_feat = criterion_kd(v_feature_s, v_feature_t)
                vloss_kd_output = criterion_output_kd(logits[-1], v_outputs_t[-1], labels=labels)

                vloss += vloss_kd_feat + vloss_kd_output

                val_loss_sum += vloss.detach()
                val_batches += 1.0

                val_kd_feat_sum += vloss_kd_feat.detach()
                val_kd_out_sum += vloss_kd_output.detach()

                preds = torch.argmax(logits[-1], dim=1)
                confmat = _update_confmat(confmat, preds, labels, args.num_classes, ignore_index=255)

        # ---- Reduce ----
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batches,  op=dist.ReduceOp.SUM)
        dist.all_reduce(confmat,      op=dist.ReduceOp.SUM)

        dist.all_reduce(val_kd_feat_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_kd_out_sum, op=dist.ReduceOp.SUM)

        val_loss_epoch = (val_loss_sum / val_batches).item()
        val_kd_feat_epoch = (val_kd_feat_sum / val_batches).item()
        val_kd_out_epoch = (val_kd_out_sum / val_batches).item()

        miou, iou_list = compute_miou_from_confmat(confmat)
        acc = compute_pixel_accuracy_from_confmat(confmat)

        aug_counts = aug_counts_local.clone()
        dist.all_reduce(aug_counts, op=dist.ReduceOp.SUM)

        # ===== Logging / Checkpoint on rank0 =====
        if local_rank == 0:
            lr_vals = scheduler.get_last_lr()
            lr = sum(lr_vals) / len(lr_vals)

            counts_str = ", ".join(f"{n}:{int(aug_counts[i].item())}" for i, n in enumerate(AUG_NAMES))
            print(f"[Epoch {epoch + 1}] Aug Applied Counts -> {counts_str}")

            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", train_loss_epoch, step)
                writer.add_scalar("val/loss",   val_loss_epoch,   step)
                writer.add_scalar("train/kd_feat", train_kd_feat_epoch, step)
                writer.add_scalar("train/kd_output", train_kd_out_epoch, step)
                writer.add_scalar("val/kd_feat", val_kd_feat_epoch, step)
                writer.add_scalar("val/kd_output", val_kd_out_epoch, step)
                writer.add_scalar("val/mIoU",   miou,             step)
                writer.add_scalar("val/Acc",    acc,              step)
                writer.add_scalar("train/lr_epoch", lr,           step)
                for c, iou_c in enumerate(iou_list):
                    if not math.isnan(iou_c):
                        writer.add_scalar(f"val/IoU_cls/{c}", iou_c, step)
                for i, n in enumerate(AUG_NAMES):
                    writer.add_scalar(f"aug/count/{n}", int(aug_counts[i].item()), step)

            with open(log_path, "a") as f:
                f.write("\n%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.8f" %
                        (epoch + 1, train_loss_epoch, val_loss_epoch, miou, acc, lr))

            # ---- Save checkpoints ----
            def save_ckpt(tag_path):
                ckpt = {
                    "model": _get_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,   # 다음 epoch(0-index)
                    "best_miou": best_miou,
                    "criterion_kd": criterion_kd.state_dict(),
                }
                if any(p.requires_grad for p in getattr(criterion_output_kd, 'parameters', lambda: [])()):
                    if any(p.requires_grad for p in criterion_output_kd.parameters()):
                        ckpt["criterion_output_kd"] = criterion_output_kd.state_dict()

                torch.save(ckpt, tag_path)

            # last (통합)
            save_ckpt(os.path.join(args.result_dir, "last.pth.tar"))

            # best by mIoU
            if (miou > best_miou + eps) or (abs(miou - best_miou) <= eps and (epoch + 1) > 0):
                best_miou  = miou
                best_epoch = epoch + 1
                # 가중치만
                torch.save(_get_state_dict(model), os.path.join(args.result_dir, "model_best.pth"))
                torch.save(_get_state_dict(model), os.path.join(args.result_dir, f"model_best_e{best_epoch}_miou{best_miou:.4f}.pth"))
                # 통합 best
                save_ckpt(os.path.join(args.result_dir, "best.pth.tar"))

        dist.barrier()

    if local_rank == 0 and writer is not None:
        writer.close()

    dist.destroy_process_group()


# ---------- Argparse ----------
if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_P2P_DISABLE"] = "1" 
    os.environ["NCCL_IB_DISABLE"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,  help="Path to dataset root",
                        default="/content/drive/MyDrive/SemanticDataset_lednet")
    parser.add_argument("--loadpath", type=str,  help="Path to dataset root", 
                        default="/content/drive/MyDrive/PIDNet_S_ImageNet.pth.tar")
    parser.add_argument("--teacher_loadpath", type=str, help="Path to dataset root",
                        default="/content/drive/MyDrive/PIDNet_L.pth")
    parser.add_argument("--resume", type=str,
                        default="/content/drive/MyDrive/PIDNet_kd/pths/last.pth.tar",
                        help="통합 ckpt(.pth/.tar) 또는 가중치(.pth) 경로")
    parser.add_argument("--resume_epoch", type=int, default=65,
                        help="weights-only 재개 시 마지막 완료 epoch(1-index)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--result_dir", type=str, default='/content/drive/MyDrive/PIDNet_kd/pths')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--crop_size", default=[1024, 1920], type=arg_as_list, help="crop size (H W)")
    parser.add_argument("--scale_range", default=[0.75, 1.25], type=arg_as_list,  help="resize Input")
    parser.add_argument("--normal_aug_prob", type=float, default=0.5, help="normal 이미지에 degradation 조합을 적용할 확률")
    parser.add_argument("--severity_min", type=int, default=1)
    parser.add_argument("--severity_max", type=int, default=5)
    
    args = parser.parse_args()
    print(f'Initial learning rate: {args.lr}')
    print(f'Total epochs: {args.epochs}')
    print(f'dataset path: {args.dataset_dir}')
                  
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn', force=True)
    train(args)
