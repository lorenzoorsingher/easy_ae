import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    def __init__(self):

        self.path = "data/lfw_funneled/"
        self.colorpath = "data/color/"
        self.bwpath = "data/bw/"

        self.data = []
        tot_files = len(os.listdir(self.colorpath))

        for i in range(tot_files):
            Xpath = self.colorpath + "color_"+str(i)+".jpg"
            ypath = self.bwpath + "bw_"+str(i)+".jpg"
            if os.path.exists(Xpath) and os.path.exists(ypath):
                self.data.append([Xpath,ypath])
                
        self.img_dim = (250, 250)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        color_path, bw_path = self.data[idx]
        Ximg = cv2.imread(bw_path)
        Ximg = cv2.cvtColor(Ximg, cv2.COLOR_BGR2GRAY)
        Ximg = cv2.resize(Ximg, self.img_dim)
        yimg = cv2.imread(color_path)
        yimg = cv2.resize(yimg, self.img_dim)
        Ximg_tensor = torch.tensor([Ximg])
        yimg_tensor = torch.tensor(yimg)
        yimg_tensor = yimg_tensor.permute(2, 0, 1)
        return Ximg_tensor, yimg_tensor