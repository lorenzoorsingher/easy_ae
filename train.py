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



BATCH = 16
EPOCH = 100
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))


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
    print("EPOCH n",i,"\n")

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
        if count%20==0:
            
            Ximg, _ = dataset.denormalize(X[0].detach().transpose(0,2).numpy(),None)
            pred, yimg = dataset.denormalize(predictions[0].detach().transpose(0,2).numpy(),y[0].detach().transpose(0,2).numpy())
            
            Ximg = Ximg.astype(np.uint8)
            Ximg = cv.resize(Ximg, yimg.shape[:2])
            Ximg = cv.cvtColor(Ximg,cv.COLOR_GRAY2BGR)
            yimg = yimg.astype(np.uint8)
            pred = pred.astype(np.uint8)

            # ex,_ = ae(Xmple)
            # Ximg2, _ = dataset.denormalize(Xmple[0].detach().transpose(0,2).numpy(),None)
            # pred2, yimg2 = dataset.denormalize(ex[0].detach().transpose(0,2).numpy(),y[0].detach().transpose(0,2).numpy())
            
            # Ximg2 = Ximg2.astype(np.uint8)
            # Ximg2 = cv.resize(Ximg2, yimg2.shape[:2])
            # Ximg2 = cv.cvtColor(Ximg2,cv.COLOR_GRAY2BGR)
            # yimg2 = yimg2.astype(np.uint8)

            #cv.imshow("imm", np.vstack([np.hstack([Ximg,yimg,pred]),np.hstack([Ximg2,yimg2,pred2])]))
            cv.imshow("imm", np.hstack([Ximg,yimg,pred]))
            #cv.imshow("imm3", np.hstack([Ximg2,yimg2,pred2]))

            print("LOSS: ", round(loss.item()/BATCH,4))
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
    
    print("loss: ", epochLoss/batchItems)