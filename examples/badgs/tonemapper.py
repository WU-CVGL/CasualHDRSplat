import torch
import torch.nn as nn
import imageio
import numpy as np


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.params = torch.nn.Parameter(torch.Tensor([1, 0]))

    def forward(self, x):
        return 1 / (1 + torch.exp(self.params[0] * (x + self.params[1])))


class ToneMapper(torch.nn.Module):
    def __init__(self, hidden=64):
        super(ToneMapper, self).__init__()
        self.Linear_r = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.Linear_g = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.Linear_b = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        # self.Linear_r = Sigmoid()
        # self.Linear_g = Sigmoid()
        # self.Linear_b = Sigmoid()
        # self.Linear_r = torch.nn.Sigmoid()
        # self.Linear_g = torch.nn.Sigmoid()
        # self.Linear_b = torch.nn.Sigmoid()

    def set_initial_values(self, imag_paths):
        images = np.array([imageio.imread(path)[..., :3] for path in imag_paths]) / 255
        linear_imgs = np.power(images, 2.2)
        overall_mean = np.mean(linear_imgs, axis=(0, 1, 2, 3))
        per_image_mean = np.mean(linear_imgs, axis=(1, 2, 3))
        ref_idx = np.argmin(np.abs(np.subtract(per_image_mean, overall_mean)))

        overall_channel_wise = np.mean(linear_imgs, axis=(0, 1, 2))
        per_image_channel_wise = np.mean(linear_imgs, axis=(1, 2))

        ratio = per_image_channel_wise / overall_channel_wise

        wb = np.log(ratio)

        return torch.Tensor(wb), ref_idx

    def setup_whitebalance(self, image_paths, image_indices, device):
        train_images_paths = np.array(image_paths)[image_indices]
        white_balance_init, train_ref_idx = self.set_initial_values(train_images_paths)
        # white_balance_init = torch.zeros((len(image_indices),3), dtype=torch.float32)
        self.wb = nn.Embedding.from_pretrained(white_balance_init, freeze=False).to(device)
        # white_balance_init, self.ref_idx = self.set_initial_values(image_paths)
        # self.wb = nn.Embedding.from_pretrained(white_balance_init, freeze=False).to(self.device)
        self.is_wb = True

        self.wb_ref_idx = image_indices[train_ref_idx]
        ref_wb = self.wb.weight[self.wb_ref_idx].detach().cpu().numpy()
        self.ref_wb = torch.exp(torch.Tensor(ref_wb)).to(device)

    def forward(self, rgb_h_ln, exps, k_times, image_ids):
        # 尝试增加一个coordinate-based query。

        # exposure_ln = self.exposure_linear(exps)
        # exposure_ln = torch.log(exps)
        exps = exps * k_times

        # if self.is_wb:
        #     if image_ids == self.wb_ref_idx:
        #         rgb_h_ln = rgb_h_ln * self.ref_wb
        #     else:
        #         modified_image_ids = image_ids - 2
        #         wb = torch.exp(self.wb.weight[modified_image_ids])
        #         rgb_h_ln = rgb_h_ln * wb

        r_ln = rgb_h_ln[:, 0:1].view(-1, 1) * exps  # [:, :, 0:1]
        g_ln = rgb_h_ln[:, 1:2].view(-1, 1) * exps  # [:, :, 1:2]
        b_ln = rgb_h_ln[:, 2:3].view(-1, 1) * exps  # [:, :, 2:3]

        # exps = torch.log(exps * k_times)
        # r_ln = rgb_h_ln[:, 0:1].view(-1,1) + exps#[:, :, 0:1]
        # g_ln = rgb_h_ln[:, 1:2].view(-1,1) + exps#[:, :, 1:2]
        # b_ln = rgb_h_ln[:, 2:3].view(-1,1) + exps#[:, :, 2:3]

        r_l = self.Linear_r(r_ln)
        g_l = self.Linear_g(g_ln)
        b_l = self.Linear_b(b_ln)
        # r_l = r_l
        # g_l = g_ln
        # b_l = b_ln
        rgb_l = torch.cat([r_l, g_l, b_l], -1)
        rgb_l = (torch.tanh(rgb_l) + 1) / 2
        # rgb_l = torch.relu(rgb_l)
        return rgb_l

    # def zero_point_contraint(self):

    def zero_point_contraint(self, gt, query_point):
        ln_x = query_point
        ln_y = query_point
        ln_z = query_point
        x_l = self.Linear_r(ln_x)
        y_l = self.Linear_g(ln_y)
        z_l = self.Linear_b(ln_z)

        rgb_l = torch.cat([x_l, y_l, z_l], -1)
        rgb_l = (torch.tanh(rgb_l) + 1) / 2
        return torch.mean((rgb_l - gt) ** 2)

    def mask_weight(self, x):
        low = 0.05
        high = 0.9

        lower = torch.where(x < low)
        higher = torch.where(x > high)
        weight = torch.ones_like(x).to(x.device)
        weight[lower] = ((weight[lower] + low) / 2 * low) ** 2
        weight[higher] = ((2 - weight[higher] / 2 * (1 - high)) ** 2)
        return weight

    @torch.no_grad()
    def export_mapping_function(self):
        N = 30

        exps = torch.arange(-N, N, N / 100).unsqueeze(1)
        r_l = self.Linear_r(exps)
        g_l = self.Linear_g(exps)
        b_l = self.Linear_b(exps)
        rgb_l = torch.cat([r_l, g_l, b_l], -1)
        rgb_l = torch.sigmoid(rgb_l)
        return exps, rgb_l
