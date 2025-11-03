import torch
import random

from scipy.io import savemat
# from Dataload import DataLoad
from torch.utils.data import DataLoader
from torch import nn, optim
from modelnet.kernel import *
import numpy as np
import os
import scipy.io as sio

device = 'cuda:2'
batch_size = 4


def spectral_sample(spectral_matrix, input):
    c, n = spectral_matrix.shape
    b, _, h, w = input.shape
    res = torch.einsum('bchw,cn->bnhw', input, spectral_matrix)
    return res


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(1)
    train_data = DataLoad(root1="./Template/modelnet/sample",
                          root2="./",
                          mode="train")
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    test_data  = DataLoad(root1="./Template/modelnet/sample",
                               root2="",
                               mode="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    model = Kernel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    criterion = nn.L1Loss()
    spectral_matrix = torch.from_numpy(sio.loadmat('./modelnet/data/data.mat')['F_h']).type(torch.float32).to(device)
    P = np.array([])
    Pn=0
    for epoch in range(1, 1001):
        train_loss = 0
        for idx, (hrMS, ref) in enumerate(train_loader):
            print('\r',idx, end="")
            hrMS = hrMS.type(torch.float32).to(device)
            hrMS = spectral_sample(spectral_matrix, hrMS)

            ref= ref.type(torch.float32).to(device)
            # lrHS = nn.functional.interpolate(lrHS, scale_factor=4, mode='bicubic')
            out = model(hrMS)
            loss = criterion(out, ref)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # savemat(f'./modelnet/hrMS/train/{idx+1}.mat', {'hrMS': hrMS.detach().cpu().numpy()})
        print('epoch', epoch, 'loss', train_loss / (idx + 1))
        if epoch % 1 == 0:
            test_loss = 0
            for idx, (lrHS, ref) in enumerate(test_loader):
                lrHS = lrHS.type(torch.float32).to(device)
                lrHS = spectral_sample(spectral_matrix, lrHS)
                ref = ref.type(torch.float32).to(device)
                out = model(lrHS)
                pn = psnr(out.cpu().detach().numpy(), ref.cpu().detach().numpy())
                P = np.append(P, pn)
                # savemat(f'./modelnet/hrMS/test/{idx + 1}.mat', {'hrMS': lrHS.detach().cpu().numpy()})
            if P.mean() > Pn:
                Pn = P.mean()
                torch.save({'model': model.state_dict()}, './modelnet/kernel/0.pth')

        print('epoch', epoch, 'test--psnr', P.mean())
