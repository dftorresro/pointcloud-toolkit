import torch, yaml, time, pathlib

def loss_fn(pred, gold, num_cls):
    pred = pred.reshape(-1, num_cls)
    gold = gold.reshape(-1)
    return torch.nn.functional.cross_entropy(pred, gold)

class Logger:
    def __init__(self, path): path.parent.mkdir(parents=True, exist_ok=True); self.f = open(path,'a')
    def log(self, txt): print(txt); self.f.write(txt+'\n'); self.f.flush()
    def close(self): self.f.close()
