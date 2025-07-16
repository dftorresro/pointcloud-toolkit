import glob, os, h5py, math, numpy as np, torch
from torch.utils.data import Dataset

def _concat_h5(partition, root):
    patt = 'train*.h5' if partition == 'train' else 'test*.h5'
    all_data, all_seg = [], []
    for h5 in glob.glob(os.path.join(root, patt)):
        with h5py.File(h5, 'r') as f:
            all_data.append(f['data'][:])  # (B, N, 3)
            all_seg.append(f['seg'][:])    # (B, N)
    return np.concatenate(all_data), np.concatenate(all_seg)

class ShapeNetPart(Dataset):
    def __init__(self, partition, num_points, root):
        self.pc, self.seg = _concat_h5(partition, root)
        self.partition, self.num_points = partition, num_points

    def __len__(self): return self.pc.shape[0]

    def __getitem__(self, idx):
        pts, lab = self.pc[idx], self.seg[idx]
        choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
        pts, lab = pts[choice], lab[choice]

        if self.partition == 'train':
            theta = np.random.uniform(0, 2*math.pi)
            rot = np.array([[math.cos(theta),-math.sin(theta),0],
                            [math.sin(theta), math.cos(theta),0],[0,0,1]])
            pts = pts @ rot
        return torch.as_tensor(pts, dtype=torch.float32), torch.as_tensor(lab, dtype=torch.long)
