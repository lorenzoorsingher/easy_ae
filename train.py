from model_files import model
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from dataLoader import CustomDataset
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
import os



BATCH = 16
EPOCH = 100
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))


dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

ae = model.SimplerAE().to(DEVICE)

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
        Ximg = X[0].numpy()
        yimg = y[0].numpy()

        #breakpoint()
        X = X.float()
        y = y.float()
        (X,y) = (X.to(DEVICE), y.to(DEVICE))
        
        predictions,_ = ae(X)

        count+=1
        if count%20==0:
            ex,_ = ae(Xmple)

            npim1 = (y[0].detach().transpose(0,2).numpy()).astype(np.uint8)
            npim2 = (predictions[0].detach().transpose(0,2).numpy()).astype(np.uint8)
            npim3 = (ymple[0].detach().transpose(0,2).numpy()).astype(np.uint8)
            npim4 = (ex[0].detach().transpose(0,2).numpy()).astype(np.uint8)
            
            cv.imshow("imm", np.hstack([npim1,npim2,npim3,npim4]))
            print("LOSS: ", round(loss.item()/BATCH,4))
            cv.waitKey(1)

            #breakpoint()
        #breakpoint()
        loss = lossFunc(predictions, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        if stop:
            #get_info(predictions,label)
            #breakpoint()
            stop = False
        epochLoss += loss.item()
        batchItems += BATCH
    
    print("loss: ", epochLoss/batchItems)