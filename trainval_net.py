import torch
import torch.utils.data
from model.faster_rcnn import faster_rcnn

## args or cfg?
CUDA = True
OPTIMIZER = "adam"
LEARNING_RATE = 0.1
START_EPOCH = 0
MAX_EPOCH = 10

def train():
    data_loader = torch.utils.data.DataLoader()
    optimizer = torch.optim.adam()
    lr = LEARNING_RATE
    model = faster_rcnn() # need fix
    if CUDA:
        model.cuda()

    for epoch in range(START_EPOCH, MAX_EPOCH + 1):
        for i, data in enumerate(data_loader):
            if CUDA:
                data.cuda()

            loss = model(data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, loss.item()))