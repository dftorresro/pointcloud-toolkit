class DGCNNPartSeg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims*3, 256, bias=False)  # adjusted dimensions to match a chosen concat scheme
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)

        self.linear2 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.linear3 = nn.Linear(256, args.num_part_classes)

    def knn(self, x, k):
        inner = -2*torch.matmul(x.transpose(2,1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2,1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx

    def get_graph_feature(self, x, k=20):
        batch_size = x.size(0)
        num_points = x.size(2)
        idx = self.knn(x, k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1,1,1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2,1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, x.size(-1))
        x = x.view(batch_size, num_points, 1, x.size(-1)).repeat(1,1,k,1)
        feature = torch.cat((feature - x, x), dim=3).permute(0,3,1,2).contiguous()
        return feature

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2,1).contiguous()

        x1 = self.get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        x2 = self.get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        x3 = self.get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        x4 = self.get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_emb = self.conv5(x_cat)
        x_max = torch.max(x_emb, 2)[0]   # [B, 1024]
        x_mean = torch.mean(x_emb, 2)    # [B, 1024]

        x_global = torch.cat((x_max, x_mean), 1)  # [B, 2048]
        x_global_expand = x_global.unsqueeze(2).repeat(1,1,x_emb.size(2)) # [B, 2048, N]

        x_seg = torch.cat((x_emb, x_global_expand), dim=1)  # [B, 3072, N]
        x_seg = x_seg.transpose(2,1).contiguous() # [B, N, 3072]
        x_seg = x_seg.view(-1, x_seg.size(-1))    # [B*N, 3072]

        x_seg = self.linear1(x_seg) # now matches dimension

        x_seg = self.bn6(x_seg)
        x_seg = torch.nn.functional.leaky_relu(x_seg, negative_slope=0.2)
        x_seg = self.dp1(x_seg)

        x_seg = self.linear2(x_seg) # (B*N, 256)
        x_seg = self.bn7(x_seg)
        x_seg = torch.nn.functional.leaky_relu(x_seg, negative_slope=0.2)
        x_seg = self.dp2(x_seg)

        x_seg = self.linear3(x_seg) # (B*N, num_part_classes)
        x_seg = x_seg.view(batch_size, args.num_point, args.num_part_classes)

        return x_seg
