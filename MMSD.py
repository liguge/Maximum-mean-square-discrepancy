import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from timm.loss import LabelSmoothingCrossEntropy     #效果远远胜过交叉熵损失
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cwru_data', type=str, default="H:\\MMSD\\48kcwru_data.npy", help='')
    parser.add_argument('--cwru_label', type=str, default="H:\\MMSD\\48kcwru_label.npy", help='')
    parser.add_argument('--bjtu_data', type=str, default="H:\\MMSD\\seu2_data.npy", help='')
    parser.add_argument('--bjtu_label', type=str, default="H:\\MMSD\\seu2_label.npy", help='')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
    parser.add_argument('--nepoch', type=int, default=200, help='max number of epoch')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--u', type=float, default=1.0, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='initialization list')
    args = parser.parse_args()
    return args


class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y

    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label

    def __len__(self):
        return len(self.Data)


def load_data():
    source_data = np.load(args.cwru_data)
    source_label = np.load(args.cwru_label).argmax(axis=-1)
    target_data = np.load(args.bjtu_data)
    target_label = np.load(args.bjtu_label).argmax(axis=-1)
    # source_data = MinMaxScaler().fit_transform(source_data.T).T
    # target_data = MinMaxScaler().fit_transform(target_data.T).T
    source_data = (source_data - source_data.min(axis=1).reshape((len(source_data), 1))) / (
                source_data.max(axis=1).reshape((len(source_data), 1)) - source_data.min(axis=1).reshape(
            (len(source_data), 1)))
    target_data = (target_data - target_data.min(axis=1).reshape((len(target_data), 1))) / (
                target_data.max(axis=1).reshape((len(target_data), 1)) - target_data.min(axis=1).reshape(
            (len(target_data), 1)))
    source_data, source_label = shuffle(source_data, source_label, random_state=0)
    target_data, target_label = shuffle(target_data, target_label, random_state=0)
    source_data = np.expand_dims(source_data, axis=1)
    target_data = np.expand_dims(target_data, axis=1)
    Train_source = Dataset(source_data, source_label)
    Train_target = Dataset(target_data, target_label)
    return Train_source, Train_target


###############################################################
class MMSD(nn.Module):
    def __init__(self):
        super(MMSD, self).__init__()

    def _mix_rbf_mmsd(self, X, Y, sigmas=(1,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmsd(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = torch.tensor(K_XX.size(0), dtype=torch.float32)
        n = torch.tensor(K_YY.size(0), dtype=torch.float32)
        C_K_XX = torch.pow(K_XX, 2)
        C_K_YY = torch.pow(K_YY, 2)
        C_K_XY = torch.pow(K_XY, 2)
        if biased:
            mmsd = (torch.sum(C_K_XX) / (m * m) + torch.sum(C_K_YY) / (n * n)
            - 2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            mmsd = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * m)
                    + (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        return mmsd

    def forward(self, X1, X2, bandwidths=[3]):
        kernel_loss = self._mix_rbf_mmsd(X1, X2, sigmas=bandwidths)
        return kernel_loss





###################################################################################################3
class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=False):
        super(MaxPool, self).__init__()
        self.padding = padding
        self.p1_1 = nn.MaxPool1d(kernel_size, stride)

    def forward(self, x):
        x = self.p1_1(x)
        if self.padding:
            x = F.pad(x, (0, x.size(-1) % self.p1_1.stride), "constant", 0)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24),
                                  nn.BatchNorm1d(16),
                                  nn.ReLU())
        self.p1_2 = MaxPool(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(16, 32, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU())
        self.p2_2 = MaxPool(2, 2)
        self.p3_1 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU())
        self.p3_2 = MaxPool(2, 2)
        self.p4_1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU())
        self.p4_2 = MaxPool(2, 2)
        self.p5_1 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU())
        self.p5_2 = MaxPool(2, 2)
        self.p6 = nn.AdaptiveAvgPool1d(1)
        self.p7_1 = nn.Sequential(nn.Linear(256, 128),
                                  nn.ReLU())
        self.p7_2 = nn.Sequential(nn.Linear(128, 4))

    def forward(self, x):
        x = self.p1_2(self.p1_1(x))
        x = self.p2_2(self.p2_1(x))
        x = self.p3_2(self.p3_1(x))
        x = self.p4_2(self.p4_1(x))
        x = self.p5_2(self.p5_1(x))
        x1 = self.p7_1(self.p6(x).squeeze())
        x2 = self.p7_2(x1)

        return x1, x2

def train(model, source_loader, target_loader, optimizer):
    lambd = -4 / (np.sqrt(epoch / (args.nepoch - epoch + 1)) + 1) + 4  #稳定的必须加  proposed by 邵海东老师 https://doi.org/10.1016/j.jmsy.2022.09.004.
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    for i in range(0, num_iter):
        source_data, source_label = next(iter_source)
        target_data, _ = next(iter_target)
        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data = target_data.cuda()
        optimizer.zero_grad()
        out1, output1 = model(source_data.float())
        out2, output2 = model(target_data.float())
        clc_loss_step = criterion(output1, source_label)
        domain_loss = criterion2(output1, output2)
        loss_step = clc_loss_step + lambd * args.u * domain_loss
        loss_step.backward()
        optimizer.step()
        metric_accuracy_1.update(output1.max(1)[1], source_label)
        metric_mean_1.update(loss_step)
        metric_mean_5.update(clc_loss_step)
        metric_mean_2.update(domain_loss)
    train_acc = metric_accuracy_1.compute()    #训练集的准确率
    train_all_loss = metric_mean_1.compute()   #整体的损失
    source_cla_loss = metric_mean_5.compute()  #分类损失
    domain_loss = metric_mean_2.compute()      #域差异损失
    metric_accuracy_1.reset()
    metric_mean_1.reset()
    metric_mean_5.reset()
    metric_mean_2.reset()
    return train_acc, train_all_loss, source_cla_loss, domain_loss

def test(model, target_loader):
    eval_loss = 0
    eval_acc = 0
    model.eval()
    iter_target = iter(target_loader)
    num_iter = len(target_loader)
    with torch.no_grad():
        for i in range(0, num_iter):
            target_data, target_label = next(iter_target)
            target_data, target_label = target_data.cuda(), target_label.cuda()
            _, output = model(target_data.float())
            # loss = criterion(output, target_label)
            # eval_loss += loss.item()
            # _, pred = output.max(1)
            # num_correct = (pred == target_label).sum().item()
            # acc = num_correct / target_data.shape[0]
            # eval_acc += acc
        # eval_losses.append(eval_loss / len(target_loader))
        # eval_acces.append(eval_acc / len(target_loader))
            metric_accuracy_2.update(output.max(1)[1], target_label)
            metric_mean_3.update(criterion(output, target_label))
        test_acc = metric_accuracy_2.compute()
        test_loss = metric_mean_3.compute()     #测试集损失
        metric_accuracy_2.reset()
        metric_mean_3.reset()
    return test_acc, test_loss

if __name__ == '__main__':
    seed_everything(3407)
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_accuracy_1 = Accuracy().cuda()
    metric_accuracy_2 = Accuracy().cuda()
    metric_mean_1 = MeanMetric().cuda()
    metric_mean_2 = MeanMetric().cuda()
    metric_mean_3 = MeanMetric().cuda()
    metric_mean_5 = MeanMetric().cuda()
    t_test_acc = 0.0
    stop = 0
    Train_source, Train_target = load_data()
    g = torch.Generator()
    source_loader = da.DataLoader(dataset=Train_source, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    target_loader = da.DataLoader(dataset=Train_target, batch_size=args.batch_size, shuffle=True, generator=g)
    target_loader_test = da.DataLoader(dataset=Train_target, batch_size=16, shuffle=False)
    model = Model().to(device)
    criterion = LabelSmoothingCrossEntropy()
    criterion2 = MMSD()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(0, args.nepoch):
        train_acc, train_all_loss, source_cla_loss, domain_loss = train(model, source_loader, target_loader, optimizer)
        # if epoch == 1:
        #     for name, parms in model.named_parameters():
        #         print('-->name:', name)
        #         print('-->para:', parms)
        #         print('-->grad_requirs:', parms.requires_grad)
        #         print('-->grad_value:', parms.grad)
        #         print("===")
        # if t_test_acc > test_acc:
        #     test_acc = t_test_acc
        #     stop = 0
        #     torch.save(model, 'model.pkl')
        test_acc, test_loss = test(model,  target_loader_test)
        print(
            'Epoch{}, test_loss is {:.5f}, train_accuracy is {:.5f},test_accuracy is {:.5f},train_all_loss is {:.5f},source_cla_loss is {:.5f},domain_loss is {:.5f}'.format(
                epoch + 1, test_loss, train_acc, test_acc, train_all_loss, source_cla_loss, domain_loss
                ))
