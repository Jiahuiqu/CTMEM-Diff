from torch.utils.data import Dataset
import os
from scipy.io import loadmat


class DataLoad(Dataset):

    def __init__(self, root, mode):
        super(DataLoad, self).__init__()
        self.root = root
        self.mode = mode

        if self.mode == "train":
            self.gtHS = os.listdir(os.path.join(self.root, "train", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, "train", "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            # self.HRMS = os.listdir(os.path.join(self.root, "unpaired"))
            self.HRMS = os.listdir(os.path.join(self.root, "train", "matched"))
            self.HRMS.sort(key=lambda x: int(x.split(".")[0]))

        elif self.mode == "test":
            self.gtHS = os.listdir(os.path.join(self.root, "test", "gtHS"))
            self.gtHS.sort(key=lambda x: int(x.split(".")[0]))
            self.LRHS = os.listdir(os.path.join(self.root, "test", "LRHS"))
            self.LRHS.sort(key=lambda x: int(x.split(".")[0]))
            # self.HRMS = os.listdir(os.path.join(self.root, "unpaired"))
            self.HRMS = os.listdir(os.path.join(self.root, "test", "matched"))
            self.HRMS.sort(key=lambda x: int(x.split(".")[0]))
            self.gtHS = self.gtHS
            self.LRHS = self.LRHS
            self.HRMS = self.HRMS
        print(self.HRMS)

    def __len__(self):
        return len(self.gtHS)

    def __getitem__(self, index):
        gt_HS, LR_HS, HR_MS = self.gtHS[index], self.LRHS[index], self.HRMS[index]
        data_ref = loadmat(os.path.join(self.root, self.mode, "gtHS", gt_HS))['patch'].transpose(2,0,1)
        data_LRHS = loadmat(os.path.join(self.root, self.mode, "LRHS", LR_HS))['patch'].transpose(2,0,1)
        data_HRMS = loadmat(os.path.join(self.root, self.mode, "matched", HR_MS))['match_img'].transpose(2,0,1)

        return data_HRMS, data_LRHS, data_ref
