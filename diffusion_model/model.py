import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from simulator import definition as armdef


class Model(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.d = 1024
        self.m = nn.ZeroPad1d((0, self.d - armdef.arm.spring_joint_count * 2))
        # 入力データに位置情報を付与し、入力データに各要素の順序を反映させる
        self.pe = PositionalEncoding(steps, self.d)
        # エンコーダーとデコーダーを定義
        self.encoder = Encoder(self.d)
        self.decoder = Decoder(self.d)
        # 全結合層を定義
        self.pos_fc = nn.Linear(2, self.d)

    def forward(self, x: torch.Tensor, step: torch.Tensor, pos: torch.Tensor):
        # #固定するインデックスと値を指定
        # fixed_indeces = [4, 5]
        # fixed_values = [0, 30]
        # #固定値を代入
        # for idx, val in zip(fixed_indeces, fixed_values):
        #     x[:,idx] = val

        # テンソルxのゼロパディングを行う
        x = self.m(x)
        # テンソルxに位置情報を付与し、入力データに各要素の順序を反映させる
        x = self.pe(x, step)
        # エンコーダーを通過させる
        x = self.encoder(x, step)
        # 入力テンソルposに対して線形変換を行い、pos_に代入
        pos_ = self.pos_fc(pos)
        # pos_を特徴量としてxに追加
        features = pos_
        x = x + features
        # デコーダ―を通過させる
        x = self.decoder(x, step)

        x = x[:, : armdef.arm.spring_joint_count * 2]
        return x

    # デノイズ処理を行う関数を定義
    def denoise(self, xt: torch.Tensor, steps: int, pos):
        denoise_time_list = []
        # ノイズを加えたときとは逆方向にデノイズ処理を行う
        for i in reversed(range(1, steps)):
            start = time.perf_counter()
            # テンソルの要素がiのテンソルを生成し、cudaメソッドを使用してGPUに転送
            step = torch.FloatTensor([i]).cuda()
            # Xtと同じサイズのテンソルを生成する
            z = torch.randn_like(xt)
            # リスト[i]を要素とするテンソルを生成し、データ型を64ビット整数に変換
            step = torch.Tensor([i]).long()
            # テンソルxt_の形状を1行の2次元テンソルに変換
            xt_ = xt.view(1, -1)
            if i == 1:
                xt = (
                    # 1/√α[i]×(xt-√β[i]×self(xt_, step, pos))
                    1 / torch.sqrt(alpha[i])
                ) * (xt - (torch.sqrt(beta[i])) * self(xt_, step, pos))
            else:
                # 1/√α[i]×(xt-β[i]/√(1-α[i])×self(xt_, step, pos)+√((1-α[i-1])/(1-α[i])×β[i]×z)
                xt = (1 / torch.sqrt(alpha[i])) * (
                    xt - (beta[i] / torch.sqrt(1 - alpha_[i])) * self(xt_, step, pos)
                ) + torch.sqrt((1 - alpha_[i - 1]) / (1 - alpha_[i]) * beta[i]) * z
            xt = xt.view(-1)
            end = time.perf_counter()
            denoise_time = end - start
            denoise_time_list.append(denoise_time)
        # デノイズ処理にかかった平均時間を計算
        print(f"Denoise time: {np.mean(denoise_time_list)} seconds")
        return xt


# β_1=10^(-4)
start_beta = 1e-4
# β_T=0.02
end_beta = 0.02
steps = 25
n = 1024
# 25要素の1次元テンソルβを生成
beta = torch.FloatTensor(steps)
# 25要素の1次元テンソルαを生成
alpha = torch.FloatTensor(steps)
# 25要素の1次元テンソルα_を生成
alpha_ = torch.FloatTensor(steps)


# βとαの値を計算する関数を定義
def pre_calc_beta_and_alpha():
    for i in range(1, steps):
        beta[i] = end_beta * ((i - 1) / (steps - 1)) + start_beta * (
            (steps - 1 - (i - 1)) / (steps - 1)
        )
        alpha[i] = 1 - beta[i]
        alpha_[i] = alpha[i]
        if i - 1 >= 1:
            alpha_[i] *= alpha_[i - 1]


pre_calc_beta_and_alpha()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)

        # 特徴量の名前を生成（例：0, 1, ..., 11）
        features = [str(i) for i in range(armdef.arm.spring_joint_count * 2)]

        # 特徴量、位置、角度を取得
        self.x = df[features].values
        self.pos = df[["x", "y"]].values
        self.theta = df["theta"].values

        # # 固定するインデックスと値
        # self.fixed_indeces = fixed_indeces
        # self.fixed_values = torch.tensor(fixed_values, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 特徴量をテンソルに変換し、必要な次元を固定
        x = torch.FloatTensor(self.x[idx]).clone()

        # 出力用の位置と角度もテンソルに
        pos = torch.FloatTensor(self.pos[idx])
        theta = torch.FloatTensor([self.theta[idx]])

        return x, pos, theta


class PositionalEncoding(torch.nn.Module):
    def __init__(self, steps, d):
        # torch.nn.Moduleの__init__メソッドを呼び出す
        super().__init__()
        # step間隔で等間隔に配置された位置情報を生成し、1次元目に追加
        pos = torch.arange(steps).unsqueeze(1)
        # 10000のd乗を計算し、divに代入
        # αは0からd-1までの偶数の値を持つテンソル
        div = torch.pow(10000, torch.arange(0, d, 2) / d)
        # 要素がsteps×dのゼロテンソルを生成
        self.pe = torch.zeros(steps, d)
        # peの偶数列を, sin(pos/div)に設定
        self.pe[:, 0::2] = torch.sin(pos / div)
        # peの奇数列を, cos(pos/div)に設定
        self.pe[:, 1::2] = torch.cos(pos / div)
        # 次元の設定
        self.d = d

    # 順伝播処理を行う関数を定義
    def forward(self, x, step):
        step = step.expand(self.d, -1).T
        pe_ = torch.gather(self.pe, 0, step.cpu()).to(x.device)
        x = x * math.sqrt(self.d) + pe_
        return x


class Encoder(nn.Module):
    def __init__(self, middle_d: int = 1024, steps: int = 25):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(middle_d, middle_d)
        self.bn1 = nn.BatchNorm1d(middle_d)
        self.fc2 = nn.Linear(middle_d, middle_d)
        self.bn2 = nn.BatchNorm1d(middle_d)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(middle_d, middle_d)
        self.bn3 = nn.BatchNorm1d(middle_d)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(middle_d, middle_d)

    # 順伝播処理を行う関数を定義
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


class Decoder(nn.Module):
    def __init__(self, middle_d: int = 1024, steps: int = 25):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(middle_d, middle_d)
        self.bn1 = nn.BatchNorm1d(middle_d)
        self.fc2 = nn.Linear(middle_d, middle_d)
        self.bn2 = nn.BatchNorm1d(middle_d)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(middle_d, middle_d)
        self.bn3 = nn.BatchNorm1d(middle_d)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(middle_d, middle_d)

    # 順伝播処理を行う関数を定義
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x


def extract(t, x_shape):
    batch_size = t.shape[0]
    out = alpha_.gather(-1, t.cpu())
    # outの形状を変更する。
    # テンソルの最初の次元のサイズをバッチサイズに設定し、残りの次元を1に設定
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# 拡散モデルの定義に基づいて、xtにノイズを加える
def gen_xt(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    # tをx0の形状に変換し、at_とする
    at_ = extract(t, x0.shape)
    # xを√at_×x0+√(1-at_)×noiseに設定
    x = torch.sqrt(at_) * x0 + torch.sqrt(1 - at_) * noise
    t = t.view(x.shape[0], 1)
    return x


# アームの出力(0~30)を正規化し、-1~1の範囲に正規化を行う関数を定義
def normalize(x: torch.Tensor):
    x -= 15
    x /= 15
    return x


# 正規化処理をもとに戻す関数を定義
def denormalize(x: torch.Tensor):
    x *= 15
    x += 15
    return x


# early stoppingの実装
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")
