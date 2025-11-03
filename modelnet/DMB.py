import torch
from torch import nn
from torch.nn import functional as F
from modelnet.kernel import Kernel
from scipy.io import loadmat
# from Dataload import DataLoad
# from torch.utils.data import DataLoader
from scipy.io import savemat
import numpy as np

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        #升维度 ori:dim=3; un1:dim=4; un2:dim=5. torch.Size[1,1,3,3]
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_gradient_nopadding(nn.Module):
    def __init__(self, device=None):
        super(Get_gradient_nopadding, self).__init__()
        self.device = device
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).to(device)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).to(device)

    def forward(self, x):
        b, c, h, w = x.shape
        x_grad = torch.tensor([]).to(self.device)
        for i in range(c):
            x_temp = x[:, i]
            x_temp_v = F.conv2d(x_temp.unsqueeze(1), self.weight_v, padding=1)
            x_temp_h = F.conv2d(x_temp.unsqueeze(1), self.weight_h, padding=1)
            x_temp = torch.sqrt(torch.pow(x_temp_v, 2) + torch.pow(x_temp_h, 2) + 1e-6)
            x_grad = torch.cat([x_grad, x_temp], dim=1)
        return x_grad


class DMB(nn.Module):

    def __init__(self, device, i=0):
        super().__init__()
        self.device = device
        self.spectral_matrix = torch.from_numpy(loadmat('../data/data.mat')['F_h']).type(torch.float32).to(device)
        self.spatial_kernel = Kernel()
        state = torch.load('../modelnet/kernel/{}.pth'.format(i))
        self.spatial_kernel.load_state_dict(state['model'])
        self.spatial_kernel = self.spatial_kernel.to(device)
        self.get_grad = Get_gradient_nopadding(device=device)

    def spectral_sample(self, input):
        c, n = self.spectral_matrix.shape
        b, _, h, w = input.shape
        res = torch.einsum('bchw,cn->bnhw', input, self.spectral_matrix)
        return res

    def similarity(self, hs, ms):
        hs = self.patch_resize(hs) # b, n1, c*ph*pw
        ms = self.patch_resize(ms) # b, n2, c*ph*pw

        similarity_matrices = hs @ ms.transpose(-1, -2)
        relative_sim = torch.softmax(similarity_matrices, dim=-1)  # (b, n1 .n2)
        soft_map = relative_sim @ ms
        hard_sim = torch.argmax(similarity_matrices, dim=-1)
        return soft_map, hard_sim

    @staticmethod
    def patch_resize(input, patch_size=16):
        b, c, h, w = input.shape
        n1 = h // patch_size
        n2 = w // patch_size
        input = input.reshape(b, c, n1, patch_size, n2, patch_size)
        input = input.permute(0, 2, 4, 1, 3, 5) # b,n,n,c,ph,pw
        input = input.reshape(b, -1, c, patch_size, patch_size).contiguous()
        return input.view(b, n1*n2, -1)

    @staticmethod
    def patch_unresize(input, patch_size=16):
        b, n, l = input.shape
        input = input.view(b, n, -1, patch_size, patch_size)
        input = input.reshape(b, int(n**0.5), int(n**0.5), input.shape[2], patch_size, patch_size)
        input = input.permute(0, 3, 1, 4, 2, 5)
        return input.reshape(b, input.shape[1], input.shape[2]*patch_size, input.shape[2]*patch_size)

    def forward(self, hs, ms):
        self.spatial_kernel = self.spatial_kernel.to(self.device)
        self.get_grad = self.get_grad.to(self.device)
        grad = self.get_grad(ms)
        hs = self.spectral_sample(hs)
        ms_d = self.spatial_kernel(ms)
        ms_diff = ms-ms_d

        soft, hard = self.similarity(hs, ms_d)

        hard_matrix = torch.cat([ms, grad, ms_diff], dim=1)
        hard_matrix = self.patch_resize(hard_matrix)
        gathered_x = torch.gather(hard_matrix, dim=1, index=hard.unsqueeze(-1).expand(-1, -1, 768*3))
        return torch.cat([self.patch_unresize(soft), self.patch_unresize(gathered_x)], dim=1)

if __name__ == "__main__":

    device = 'cuda:2'
    model = DMB('cuda:2').to(device)
    # x = torch.randn(4, 102, 160, 160).to(device)
    # y = torch.randn(4, 4, 592, 320).to(device)
    #
    # print(model(x, y).shape)

    batch_size = 1
    train_data = DataLoad("", mode="train")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    test_data = DataLoad("", mode="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    for idx, (hrMS, lrHS, ref) in enumerate(train_loader):
        lrHS = lrHS.type(torch.float).to(device)
        ref = ref.type(torch.float32).to(device)
        # print(lrHS.shape)
        train_data = F.interpolate(lrHS, scale_factor=4, mode='bicubic')
        print(train_data.shape)
        train_data = model.spectral_sample(train_data)
        savemat('./sample/train/{}.mat'.format(idx + 1), {'lrMS': np.array(train_data.detach().squeeze().cpu().numpy())})

    for idx, (hrMS, lrHS, ref) in enumerate(test_loader):
        lrHS = lrHS.type(torch.float).to(device)
        ref = ref.type(torch.float32).to(device)
        train_data = F.interpolate(lrHS, scale_factor=4, mode='bicubic')
        print(train_data.shape)
        train_data = model.spectral_sample(train_data)
        savemat('./sample/test/{}.mat'.format(idx+1), {'lrMS': np.array(train_data.detach().squeeze().cpu().numpy())})
