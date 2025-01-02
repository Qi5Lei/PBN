import random
import numpy as np
import torch
import torch.nn as nn

__all__ = ['block_split_random_block_bn2d_two_type_channel_fusion_close_affine']


class _BlockBN(nn.Module):
    def __init__(self, batchnorm, num_norms=1, target_idx=-1, point_group=1, cfg=None, block_bn_idx=None):
        super(_BlockBN, self).__init__()
        self._check_bn_type(batchnorm)
        # for random select point to normalize in point_group_num groups
        self.point_group = point_group
        # for random select blocks to normalize in block_bn_idx groups
        self.block_bn_idx = block_bn_idx
        self.cfg = cfg
        self.target_idx = target_idx
        assert num_norms > 0, "The number of domains should be > 0!"
        self.num_norms = num_norms
        self.bns = batchnorm

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, inputs):
        raise NotImplementedError

    def _check_bn_type(self, batchnorm):
        raise NotImplementedError

    def forward(self, x):
        self._check_input_dim(x)
        if self.training:
            return self._forward_train(x)
        else:
            return self._forward_test(x)

    def _forward_train(self, x):
        bs = x.size(0)
        assert bs % self.num_norms == 0
        bs = int(bs // self.num_norms)
        split = torch.split(x, bs)
        out = []
        for idx, subx in enumerate(split):
            assert subx.size(0) > 0
            out.append(self.bns[idx](subx.contiguous()))
        return torch.cat(out, 0)

    def _forward_test(self, x):
        # Default: the last BN is adopted for target domain
        return self.bns[self.target_idx](x)


class _BlockBN2d(_BlockBN):
    def _check_bn_type(self, batchnorm):
        pass

    def _check_input_dim(self, inputs):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(inputs.dim()))


class block_split_random_block_bn2d_two_type_channel_fusion_close_affine(_BlockBN2d):
    def __init__(self, batchnorm, *args, **kwargs):
        super(block_split_random_block_bn2d_two_type_channel_fusion_close_affine, self).__init__(batchnorm, *args,
                                                                                                 **kwargs)

    def _forward_train(self, x):

        assert len(x.size()) == 4, 'the size of x != 4'
        assert x.shape[2] > 1, 'h == 1'
        assert x.shape[3] > 1, 'w == 1'
        res = []
        idx_all = list(np.arange(3))
        idx_sample = random.sample(idx_all, 2)
        for i in range(2):
            rand_idx = idx_sample[i]
            if rand_idx == 0:
                # baseline
                res.append(self.bns(x.contiguous(), close_affine=(i == 1)))
            elif rand_idx == 1:
                # two block
                rand_two = np.random.randint(2)
                if rand_two == 0:
                    res.append(self._forward_two_random_l_r_block(x, close_affine=(i == 1)))
                else:
                    res.append(self._forward_two_random_u_d_block(x, close_affine=(i == 1)))
            else:
                res.append(self._forward_four_random_block(x, close_affine=(i == 1)))

        s = torch.from_numpy(
            np.random.binomial(n=1, p=0.5, size=x.shape[1]).reshape(1, x.shape[1], 1,
                                                                    1)).float().cuda()
        return (1 - s) * res[0] + s * res[1]

    def _forward_two_random_u_d_block(self, x, close_affine):
        assert len(x.size()) == 4, 'the size of x != 4'
        h = x.shape[2]
        assert h > 4, 'h <= 4'
        row_split = np.random.randint(h // 4, h // 4 * 3)
        x_u, x_d = torch.split(x, [row_split, h - row_split], dim=2)
        x_input = [x_u, x_d]
        out_f = []  # x_lu_feature    x_ru_feature    x_ld_feature    x_rd_feature
        for idx, x in enumerate(x_input):
            out_f.append(self.bns(x.contiguous(), close_affine=close_affine))
        return torch.cat((out_f[0], out_f[1]), dim=2)

    def _forward_two_random_l_r_block(self, x, close_affine):
        assert len(x.size()) == 4, 'the size of x != 4'
        w = x.shape[3]
        assert w > 4, 'w <= 4'
        col_split = np.random.randint(w // 4, w // 4 * 3)
        x_l, x_r = torch.split(x, [col_split, w - col_split], dim=3)
        x_input = [x_l, x_r]
        out_f = []  # x_lu_feature    x_ru_feature    x_ld_feature    x_rd_feature
        for idx, x in enumerate(x_input):
            out_f.append(self.bns(x.contiguous(), close_affine=close_affine))
        return torch.cat((out_f[0], out_f[1]), dim=3)

    def _forward_four_random_block(self, x, close_affine):
        assert len(x.size()) == 4, 'the size of x != 4'
        h, w = x.shape[2], x.shape[3]
        assert h > 4, 'h <= 4'
        assert w > 4, 'w <= 4'
        row_split = np.random.randint(h // 4, h // 4 * 3)
        col_split = np.random.randint(w // 4, w // 4 * 3)
        x_u, x_d = torch.split(x, [row_split, h - row_split], dim=2)
        x_lu, x_ru = torch.split(x_u, [col_split, w - col_split], dim=3)
        x_ld, x_rd = torch.split(x_d, [col_split, w - col_split], dim=3)
        x_input = [x_lu, x_ru, x_ld, x_rd]
        out_f = []  # x_lu_feature    x_ru_feature    x_ld_feature    x_rd_feature
        for idx, x in enumerate(x_input):
            out_f.append(self.bns(x.contiguous(), close_affine=close_affine))
        return torch.cat((torch.cat((out_f[0], out_f[1]), dim=3),
                          torch.cat((out_f[2], out_f[3]), dim=3)), dim=2)

    def _forward_nine_random_block(self, x, close_affine):
        assert len(x.size()) == 4, 'the size of x != 4'
        h, w = x.shape[2], x.shape[3]
        assert h > 4, 'h <= 4'
        assert w > 4, 'w <= 4'
        row_split_one = np.random.randint(h // 4, h // 2)
        row_split_two = np.random.randint(h // 2, h * 3 // 4)
        col_split_one = np.random.randint(w // 4, w // 2)
        col_split_two = np.random.randint(w // 2, w * 3 // 4)
        x_u, x_m, x_d = torch.split(x, [row_split_one, row_split_two - row_split_one, h - row_split_two], dim=2)
        x_lu, x_mu, x_ru = torch.split(x_u, [col_split_one, col_split_two - col_split_one, w - col_split_two], dim=3)
        x_lm, x_mm, x_rm = torch.split(x_m, [col_split_one, col_split_two - col_split_one, w - col_split_two], dim=3)
        x_ld, x_md, x_rd = torch.split(x_d, [col_split_one, col_split_two - col_split_one, w - col_split_two], dim=3)
        x_input = [x_lu, x_mu, x_ru, x_lm, x_mm, x_rm, x_ld, x_md, x_rd]
        out_f = []  # x_lu_feature    x_ru_feature    x_ld_feature    x_rd_feature
        for idx, x in enumerate(x_input):
            out_f.append(self.bns(x.contiguous(), close_affine=close_affine))
        return torch.cat((torch.cat((out_f[0], out_f[1], out_f[2]), dim=3),
                          torch.cat((out_f[3], out_f[4], out_f[5]), dim=3),
                          torch.cat((out_f[6], out_f[7], out_f[8]), dim=3)), dim=2)

    def _forward_test(self, x):
        return self.bns(x.contiguous())

