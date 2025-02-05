import torch
import torch.nn as nn
from simulator import definition as armdef
import numpy as np
import pandas as pd
import math

#付加的な情報を付け加えるためのMiddleLayerを定義
class MiddleLayer(nn.Module):
    def __init__(self, target_d, d):
        super().__init__()
        self.encoder = FC(d)
        self.decoder = FC(d)
        self.fc = nn.Linear(target_d, d)

    def forward(self, x, target):
        x = self.encoder(x)
        x = x + self.fc(target)
        x = self.decoder(x)
        return x

class Model(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.d = 1024

        #self.m = nn.ZeroPad1d(
        #    (0, self.d - armdef.arm.spring_joint_count*2))

        self.pe = PositionalEncoding(steps, self.d)
        self.middles = nn.ModuleList([])
        self.fc = nn.Linear(armdef.arm.spring_joint_count*2, self.d)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, step: torch.Tensor, feature: torch.Tensor):
        x = self.fc(x)
        x = self.relu(x)
        x = self.pe(x, step)
        for middle in self.middles:
            x = middle(x, feature)
        return x

    def denoise(self, xt: torch.Tensor, steps: int, pos):
        for i in reversed(range(1, steps)):
            step = torch.FloatTensor([i]).cuda()
            z = torch.randn_like(xt)
            step = torch.Tensor([i]).long()
            xt_ = xt.view(1, -1)
            if i == 1:
                xt = (
                    1/torch.sqrt(alpha[i]))*(xt - (torch.sqrt(beta[i]))*self(xt_, step, pos))
            else:
                xt = (1/torch.sqrt(alpha[i]))*(xt-(beta[i]/torch.sqrt(1-alpha_[i]))*self(
                    xt_, step, pos))+torch.sqrt((1-alpha_[i-1])/(1-alpha_[i])*beta[i])*z
            xt = xt.view(-1)
        return xt

    def denoise_once(self, xt: torch.Tensor, i: int, pos):
        step = torch.FloatTensor([i]).cuda()
        z = torch.randn_like(xt)
        step = torch.Tensor([i]).long()
        xt_ = xt.view(1, -1)
        if i == 1:
            xt = (
                1/torch.sqrt(alpha[i]))*(xt - (torch.sqrt(beta[i]))*self(xt_, step, pos))
        else:
            xt = (1/torch.sqrt(alpha[i]))*(xt-(beta[i]/torch.sqrt(1-alpha_[i]))*self(
                xt_, step, pos))+torch.sqrt((1-alpha_[i-1])/(1-alpha_[i])*beta[i])*z
        xt = xt.view(-1)
        print("after denoise_once:",xt)
        return xt

class ModelForXY(Model):
    def __init__(self, steps):
        super().__init__(steps)
        self.middles.append(MiddleLayer(2, self.d))

class ModelForTheta(Model):
    def __init__(self, steps):
        super().__init__(steps)
        self.middles.append(MiddleLayer(1, self.d))

class ControlNet(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.d = 1024
        self.pos_d = 2
        self.theta_d = 1

        #self.m = nn.ZeroPad1d(
        #    (0, self.d - armdef.arm.spring_joint_count*2))
        self.fc = nn.Linear(armdef.arm.spring_joint_count*2, self.d)
        self.relu = nn.ReLU()

        self.pe = PositionalEncoding(steps, self.d)

        self.encoder = FC(self.d)
        self.decoder = FC(self.d)

        #self.m_copy = nn.ZeroPad1d(
        #    (0, self.d - armdef.arm.spring_joint_count*2))
        self.fc_copy = nn.Linear(armdef.arm.spring_joint_count*2, self.d)
        self.relu_copy = nn.ReLU()

        self.pe_copy = PositionalEncoding(steps, self.d)

        self.encoder_copy = FC(self.d)
        self.decoder_copy = FC(self.d)

        self.fc1 = nn.Linear(self.pos_d, self.d)
        self.fc1_copy = nn.Linear(self.pos_d, self.d)
        self.fc2 = nn.Linear(self.theta_d, self.d)
        self.zeroconv1 = nn.Conv1d(1, 1, 1)
        self.zeroconv2 = nn.Conv1d(1, 1, 1)
        self.zeroconv3 = nn.Conv1d(1,1,1)
        nn.init.zeros_(self.zeroconv1.weight)
        nn.init.zeros_(self.zeroconv2.weight)
        nn.init.zeros_(self.zeroconv3.weight)
        self.last_fc = nn.Linear(self.d, armdef.arm.spring_joint_count*2)

    def forward(self, x: torch.Tensor, step: torch.Tensor, feature1: torch.Tensor, feature2: torch.Tensor = None):

        if feature2 is None:
            self.fc.requires_grad_(True)
            self.relu.requires_grad_(True)
            self.pe.requires_grad_(True)
            self.encoder.requires_grad_(True)
            self.fc1.requires_grad_(True)
            self.decoder.requires_grad_(True)
            self.last_fc.requires_grad_(True)

            x = self.fc(x)
            x = self.relu(x)
            x = self.pe(x, step)
            x = self.encoder(x)
            x = x + self.fc1(feature1)
            x = self.decoder(x)
            x = self.last_fc(x)
            return x
        else:
            self.fc.requires_grad_(False)
            self.relu.requires_grad_(False)
            self.pe.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.fc1.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.last_fc.requires_grad_(False)

            x = self.fc(x)
            x = self.relu(x)
            x_first = x
            x = self.pe(x, step)
            x = self.encoder(x)
            x = x + self.fc1(feature1)

            x_ = self.fc2(feature2)
            x_ = self.zeroconv1(x_.view(-1,1,1024)).view(-1,1024)
            x_ = x_ + x_first
            x_ = self.pe_copy(x_, step)
            x_ = self.encoder_copy(x_)
            x_ = x_ + self.fc1_copy(feature1)
            x__ = x_
            x_ = self.zeroconv2(x_.view(-1,1,1024)).view(-1,1024)
            x = self.decoder(x + x_)
            x_ = self.decoder_copy(x__)
            x = x + self.zeroconv3(x_.view(-1,1,1024)).view(-1, 1024)
            x = self.last_fc(x)
            return x

    def denoise(self, xt: torch.Tensor, steps: int, pos, theta):
        for i in reversed(range(1, steps)):
            step = torch.FloatTensor([i]).cuda()
            z = torch.randn_like(xt)
            step = torch.Tensor([i]).long()
            xt_ = xt.view(1, -1)
            if i == 1:
                xt = (
                    1/torch.sqrt(alpha[i]))*(xt - (torch.sqrt(beta[i]))*self(xt_, step, pos, theta))
            else:
                xt = (1/torch.sqrt(alpha[i]))*(xt-(beta[i]/torch.sqrt(1-alpha_[i]))*self(
                    xt_, step, pos,theta))+torch.sqrt((1-alpha_[i-1])/(1-alpha_[i])*beta[i])*z
            xt = xt.view(-1)
        return xt

    def denoise_once(self, xt: torch.Tensor, i: int, pos, theta):
        step = torch.FloatTensor([i]).cuda()
        z = torch.randn_like(xt)
        step = torch.Tensor([i]).long()
        xt_ = xt.view(1, -1)
        if i == 1:
            xt = (
                1/torch.sqrt(alpha[i]))*(xt - (torch.sqrt(beta[i]))*self(xt_, step, pos, theta))
        else:
            xt = (1/torch.sqrt(alpha[i]))*(xt-(beta[i]/torch.sqrt(1-alpha_[i]))*self(
                xt_, step, pos, theta))+torch.sqrt((1-alpha_[i-1])/(1-alpha_[i])*beta[i])*z
        xt = xt.view(-1)
        return xt


start_beta = 1e-4
end_beta = 0.02
steps = 25
n = 1024

beta = torch.FloatTensor(steps)
alpha = torch.FloatTensor(steps)
alpha_ = torch.FloatTensor(steps)


def pre_calc_beta_and_alpha():
    for i in range(1, steps):
        beta[i] = end_beta*((i-1)/(steps-1))+start_beta * \
            ((steps-1-(i-1))/(steps-1))
        alpha[i] = 1-beta[i]
        alpha_[i] = alpha[i]
        if i-1 >= 1:
            alpha_[i] *= alpha_[i-1]

pre_calc_beta_and_alpha()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        features = ([str(i) for i in range(armdef.arm.spring_joint_count*2)])
        self.pos = df[['x', 'y']].values
        self.x = df[features].values
        self.theta = df['theta'].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        pos = torch.FloatTensor(self.pos[idx])
        theta = torch.FloatTensor([self.theta[idx]])
        return x, pos, theta


class PositionalEncoding(torch.nn.Module):
    def __init__(self, steps, d):
        super().__init__()
        pos = torch.arange(steps).unsqueeze(1)
        div = torch.pow(10000, torch.arange(0, d, 2)/d)
        self.pe = torch.zeros(steps, d)
        self.pe[:, 0::2] = torch.sin(pos/div)
        self.pe[:, 1::2] = torch.cos(pos/div)
        self.d = d

    def forward(self, x, step):
        step = step.expand(self.d, -1).T
        pe_ = torch.gather(self.pe, 0, step.cpu()).to(x.device)
        x = x*math.sqrt(self.d)+pe_
        return x


class FC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, d)
        self.bn2 = nn.BatchNorm1d(d)
        self.fc3 = nn.Linear(d, d)
        self.bn3 = nn.BatchNorm1d(d)
        self.fc4 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


def extract(t, x_shape):
    batch_size = t.shape[0]
    out = alpha_.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,)*(len(x_shape) - 1))).to(t.device)


def gen_xt(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    at_ = extract(t, x0.shape)
    x = torch.sqrt(at_)*x0+torch.sqrt(1-at_)*noise
    t = t.view(x.shape[0], 1)
    return x


def normalize(x: torch.Tensor):
    x -= 20
    x /= 20
    return x


def denormalize(x: torch.Tensor):
    x *= 20
    x += 20
    return x
