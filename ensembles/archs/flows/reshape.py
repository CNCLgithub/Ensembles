class ChannelSplit(Flow):
    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        log_det = 0
        return z, log_det

    def invert(self):
        return ChannelMerge()


class ChannelMerge(Flow):
    def forward(self, z1, z2):
        z = torch.cat([z1, z2], 1)
        log_det = 0
        return z, log_det

    def invert(self):
        return ChannelSplit()

class CheckerboardSplit(Flow):
    def forward(self, z):
        n_dims = z.dim()
        cb0 = 0
        cb1 = 1
        for i in range(1, n_dims):
            cb0_ = cb0
            cb1_ = cb1
            cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
            cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
        cb = cb1 if 'inv' in self.mode else cb0
        cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
        cb = cb.to(z.device)
        z_size = z.size()
        z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        log_det = 0
        return z1, z2, log_det

    def invert(self):
        return CheckboardMerge()


class CheckerboardMerge(Flow):
    def forward(self, z1, z2):
        n_dims = z1.dim()
        z_size = list(z1.size())
        z_size[-1] *= 2
        cb0 = 0
        cb1 = 1
        for i in range(1, n_dims):
            cb0_ = cb0
            cb1_ = cb1
            cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
            cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
        cb = cb1 if 'inv' in self.mode else cb0
        cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
        cb = cb.to(z1.device)
        z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
        z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
        z = cb * z1 + (1 - cb) * z2
        log_det = 0
        return z, log_det

    def invert(self):
        return CheckboardMerge()
