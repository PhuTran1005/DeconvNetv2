from models import deconvnet_v2
from utils import metrics
from utils import losses

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_toolbelt import losses as L


class DeconvNetv2():
    def __init__(self, loss, num_classes=21, init_weights=True, ignore_index=-1, gpu_id=0, print_freq=10, epoch_print=10, is_sk=True, is_attent=True):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print

        torch.cuda.set_device(self.gpu)

        self.get_loss_func = losses.get_loss(loss)
        if self.get_loss_func == "CRFLoss":
            self.loss_function = losses.CRFLoss().cuda(self.gpu)
        if self.get_loss_func == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index).cuda(self.gpu)
        if self.get_loss_func == "dice_loss":
            self.loss_function = L.DiceLoss(ignore_index=255)
        if self.get_loss_func == "soft_dice":
            self.loss_function = losses.SoftDiceLoss().cuda(self.gpu)
        if self.get_loss_func == "focal":
            self.loss_function = L.FocalLoss(ignore_index=255)
        if self.get_loss_func == "log_cosh_dice":
            self.loss_function = losses.LogCoshDiceLoss().cuda(self.gpu)

        self.model = deconvnet_v2.DeconvNetv2(self.num_classes, init_weights, is_sk, is_attent).cuda(self.gpu)

        self.eps = 1e-10
        self.best_mIoU = 0.


    def train(self, train_data, test_data, save=False, epochs=74, lr=0.01, momentum=0.9, weight_decay=0.0005):
        
        optimizer = optim.SGD(self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

        self.model.train()
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0:
                print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in enumerate(train_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)
                if self.get_loss_func == "CRFLoss":
                    loss = self.loss_function(output, y, X)
                if self.get_loss_func in ["cross_entropy", "dice_loss", "soft_dice", "focal", "log_cosh_dice"]:
                    loss = self.loss_function(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % self.print_freq == 0:
                    test_mIoU, test_loss = self.test(test_data)
                    
                    if epoch % self.epoch_print == 0:
                        state = ('Iteration : {} - Train Loss : {:.6f}, Test Loss : {:.6f}, '
                                 'Test mIoU : {:.4f}').format(i+1, loss.item(), test_loss, 100 * test_mIoU)
                        if test_mIoU > self.best_mIoU:
                            print()
                            print('*' * 35, 'Best mIoU Updated', '*' * 35)
                            print(state)
                            self.best_mIoU = test_mIoU
                            if save:
                                torch.save(self.model.state_dict(), './best.pt')
                                print('Saved Best Model')
                            print()
                        else:
                            print(state)


    def test(self, test_data):
        # only consider for batch size 1 on test_data
        tps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        fps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        fns = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        losses = list()

        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.model(X)

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                tp, fp, fn = metrics.mIoU(output, y, self.num_classes, self.gpu, ignore_index=255)
                tps += tp
                fps += fp
                fns += fn
        self.model.train()
        mIoU = torch.sum(tps / (self.eps + tps + fps + fns)) / self.num_classes
        return (mIoU.item(), sum(losses)/len(losses))