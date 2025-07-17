import torch, typer, pathlib, random, numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pcseg.data.shapenet_part import ShapeNetPart
from pcseg.models.dgcnn import DGCNNPartSeg
from pcseg.engine.utils import loss_fn, Logger

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return device

def train_loop(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = set_seed(cfg.misc.seed)

    log_root = pathlib.Path(cfg.log_dir) / time.strftime("%Y-%m-%d_%H-%M-%S")
    logger = Logger(log_root / 'log.txt')
    logger.log("🟢 Training started")

    train_ds = ShapeNetPart('train', cfg.dataset.num_points, cfg.dataset.root)
    test_ds  = ShapeNetPart('test',  cfg.dataset.num_points, cfg.dataset.root)
    train_ld = DataLoader(train_ds, cfg.optim.batch_size, True,  num_workers=cfg.misc.num_workers, drop_last=True)
    test_ld  = DataLoader(test_ds,  cfg.optim.batch_size, False, num_workers=cfg.misc.num_workers)

    model = DGCNNPartSeg(k=cfg.model.k, emb_dims=cfg.model.emb_dims,
                         dropout=cfg.model.dropout, num_part_classes=cfg.dataset.num_part_classes).to(device)
    logger.log(f"Model: {model.__class__.__name__}  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    opt = (torch.optim.SGD if cfg.optim.use_sgd else torch.optim.Adam)(
        model.parameters(), lr=cfg.optim.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.optim.step_size, cfg.optim.gamma) \
            if cfg.optim.scheduler == 'step' else None

    best = 0.0
    for epoch in range(1, cfg.optim.epochs+1):
        # ---- train
        model.train(); loss_sum=correct=pts=0
        for x,y in train_ld:
            x,y = x.cuda(), y.cuda()
            opt.zero_grad(); pred = model(x); loss = loss_fn(pred,y,cfg.dataset.num_part_classes)
            loss.backward(); opt.step()
            loss_sum += loss.item(); correct += (pred.argmax(2)==y).sum().item(); pts += y.numel()
        logger.log(f"Epoch {epoch} train  loss {loss_sum/len(train_ld):.4f}  acc {correct/pts:.4f}")

        # ---- test
        model.eval(); loss_sum=correct=pts=0
        with torch.no_grad():
            for x,y in test_ld:
                x,y = x.cuda(), y.cuda()
                pred = model(x);  loss = loss_fn(pred,y,cfg.dataset.num_part_classes)
                loss_sum += loss.item(); correct += (pred.argmax(2)==y).sum().item(); pts += y.numel()
        acc = correct/pts
        logger.log(f"Epoch {epoch} test   loss {loss_sum/len(test_ld):.4f}  acc {acc:.4f}")

        if acc > best:
            best = acc
            torch.save(model.state_dict(), log_root / 'best_model.pth')
        if sched: sched.step()
    logger.log(f"✅ Finished. Best accuracy {best:.4f}")
