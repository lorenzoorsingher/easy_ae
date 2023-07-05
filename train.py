from model_files import model
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from dataLoader import CustomDataset
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, L1Loss
from torch.optim import SGD
import os
from copy import copy
import time

SAVE_PATH = "checkpoints/"
VISUAL = 20
BATCH = 16
EPOCH = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

current_savepath = SAVE_PATH + "run_"+str(round(time.time()))+"/"
os.mkdir(current_savepath)

dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

ae = model.SimplerAE2().to(DEVICE)

model_parameters = filter(lambda p: p.requires_grad, ae.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params, " total params")
print(ae)

opt = SGD(ae.parameters(), lr=LR)
lossFunc = nn.MSELoss()

Xmple , ymple = next(iter(data_loader))   
Xmple = Xmple.float()
ymple= ymple.float()

ae.train()

cv.namedWindow("imm", cv.WINDOW_NORMAL)
for i in range(EPOCH):
    print("############## EPOCH n",i,"\n")

    epochLoss = 0
    batchItems = 0
    stop = True
    count = 0
    for batch_id, (X,y) in enumerate(data_loader):

        X = X.float()
        y = y.float()
        (X,y) = (X.to(DEVICE), y.to(DEVICE))
        
        predictions,_ = ae(X)
        count+=1
        if count%VISUAL==0:
            
            Ximg, _ = dataset.denormalize(copy(X[0].detach().transpose(0,2).numpy()),None)
            pred, yimg = dataset.denormalize(copy(predictions[0].detach().transpose(0,2).numpy()),copy(y[0].detach().transpose(0,2).numpy()))
            
            Ximg = Ximg.astype(np.uint8)
            Ximg = cv.resize(Ximg, yimg.shape[:2])
            Ximg = cv.cvtColor(Ximg,cv.COLOR_GRAY2BGR)
            yimg = yimg.astype(np.uint8)
            pred = pred.astype(np.uint8)

            example_pred, _ = ae(Xmple)
            ex_Ximg, _ = dataset.denormalize(copy(Xmple[0].detach().transpose(0,2).numpy()),None)
            ex_pred, ex_yimg = dataset.denormalize(copy(example_pred[0].detach().transpose(0,2).numpy()),copy(ymple[0].detach().transpose(0,2).numpy()))

            ex_Ximg = ex_Ximg.astype(np.uint8)
            ex_Ximg = cv.resize(ex_Ximg, yimg.shape[:2])
            ex_Ximg = cv.cvtColor(ex_Ximg,cv.COLOR_GRAY2BGR)
            ex_yimg = ex_yimg.astype(np.uint8)
            ex_pred = ex_pred.astype(np.uint8)

            cv.imshow("imm", np.vstack([np.hstack([Ximg,yimg,pred]),np.hstack([ex_Ximg,ex_yimg,ex_pred])]))
            #print("LOSS: ", round((loss.item()*BATCH)/VISUAL,4))
            cv.waitKey(1)

        loss = lossFunc(predictions, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        if count%100==0 and stop:
            #get_info(predictions,label)
            #breakpoint()
            stop = False

        
        epochLoss += loss.item()
        batchItems += BATCH
    print("[SAVE] saving checkpoint...")
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': ae.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': epochLoss/batchItems,
            }, current_savepath + "checkpoint_"+str(i)+".chkp")
    print("loss: ", epochLoss/batchItems)