from model_files import model
import numpy as np
import cv2 as cv
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

ae = model.SAE().to(DEVICE)

model_parameters = filter(lambda p: p.requires_grad, ae.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params, " total params")
print(ae)

im = cv.imread("data/bw/bw_9.jpg")

gim = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

tgim = torch.tensor([[gim]]).float()

#breakpoint()

out = ae(tgim)

breakpoint()

cv.imshow("im",im)
cv.waitKey(0)