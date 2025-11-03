import torch
import random
from Dataload import DataLoad
from torch.utils.data import DataLoader
from scipy.io import savemat
import os
from metrics_utils import *
from DDPM.ddpm import DDPM

device = 'cuda:1'
batch_size = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# cpu_num = 2  # 获取最大cpu核心数目
# os.environ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# torch.set_num_threads(cpu_num)

if __name__ == "__main__":
    set_seed(1)
    train_data = DataLoad("", mode="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_data = DataLoad("", mode="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    test_loader = iter(test_loader)
    ddpm = DDPM(device=device, image_size=160, channels=194, loss_type='l1', conditional=True, load_net=False)

    pmax = 0
    print(len(test_loader))
    for epoch in range(1, 1001):
        mean_loss = 0
        for idx, (hrMS, lrHS, ref) in enumerate(train_loader):
            print("\r", idx, end=" ")
            hrMS = hrMS.type(torch.float).to(device)
            lrHS = lrHS.type(torch.float).to(device)
            ref = ref.type(torch.float).to(device)
            loss = ddpm.optimize_parameters(lrHS, hrMS, ref)
            mean_loss += loss.item()
        print('epoch', epoch, 'loss', mean_loss / (idx + 1))
        if epoch % 50 == 0:
            ddpm.save_network(epoch=epoch)
            P = np.array([])
            for idx,(hrMS, lrHS, ref) in enumerate(test_loader):
                hrMS = hrMS.type(torch.float).to(device)
                lrHS = lrHS.type(torch.float).to(device)
                ref = ref.type(torch.float).to(device)
                ddpm.test(lrHS, hrMS, ref)
                pn = psnr(ddpm.SR.cpu().detach().numpy(), ref.cpu().detach().numpy())
                savemat('./result/{}.mat'.format(idx+1), {'out':np.array(ddpm.SR.cpu().detach().numpy()), 'ref': np.array(ref.cpu().detach().numpy())})
            print('test--idx', idx, 'psnr', pn)
            print('epoch', epoch, 'test--psnr', P.mean())


