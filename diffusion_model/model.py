import torch
import torch.nn as nn
from simulator import definition as armdef
import numpy as np
import pandas as pd
import math


class Model(nn.Module):
    def __init__(self, steps):
        super().__init__()
        #次元を1024に設定
        self.d = 1024
        #ZeroPad1dメソッドを使用して、(0, self.d - armdef.arm.spring_joint_count*2)の範囲でゼロパディングを行う.
        self.m = nn.ZeroPad1d(
            (0, self.d - armdef.arm.spring_joint_count*2))
        #入力データに位置情報を付与し、入力データに各要素の順序を反映させる
        self.pe = PositionalEncoding(steps, self.d)
        #エンコーダーとデコーダーを定義
        self.encoder = FC(self.d)
        self.decoder = FC(self.d)
        #全結合層を定義
        self.pos_fc = nn.Linear(2, self.d)

    def forward(self, x: torch.Tensor, step: torch.Tensor, pos: torch.Tensor):
        #テンソルxのゼロパディングを行う
        x = self.m(x)
        #テンソルxに位置情報を付与し、入力データに各要素の順序を反映させる
        x = self.pe(x, step)
        #エンコーダーを通過させる
        x = self.encoder(x)
        #入力テンソルposに対して線形変換を行い、pos_に代入
        pos_ = self.pos_fc(pos)
        #pos_を特徴量としてxに追加
        features = pos_
        x = x + features
        #デコーダ―を通過させる
        x = self.decoder(x)
        
        x = x[:, :armdef.arm.spring_joint_count*2]
        return x
    #デノイズ処理を行う関数を定義
    def denoise(self, xt: torch.Tensor, steps: int, pos):
        #ノイズを加えたときとは逆方向にデノイズ処理を行う
        for i in reversed(range(1, steps)):
            #テンソルの要素がiのテンソルを生成し、cudaメソッドを使用してGPUに転送
            step = torch.FloatTensor([i]).cuda()
            #Xtと同じサイズのテンソルを生成する
            z = torch.randn_like(xt)
            #リスト[i]を要素とするテンソルを生成し、データ型を64ビット整数に変換
            step = torch.Tensor([i]).long()
            #テンソルxt_の形状を1行の2次元テンソルに変換
            xt_ = xt.view(1, -1)
            if i == 1:
                xt = (
                    #1/√α[i]×(xt-√β[i]×self(xt_, step, pos))   
                    1/torch.sqrt(alpha[i]))*(xt - (torch.sqrt(beta[i]))*self(xt_, step, pos))
            else:
                #1/√α[i]×(xt-β[i]/√(1-α[i])×self(xt_, step, pos)+√((1-α[i-1])/(1-α[i])×β[i]×z)
                xt = (1/torch.sqrt(alpha[i]))*(xt-(beta[i]/torch.sqrt(1-alpha_[i]))*self(
                    xt_, step, pos))+torch.sqrt((1-alpha_[i-1])/(1-alpha_[i])*beta[i])*z
            xt = xt.view(-1)
        return xt

#β_1=10^(-4)
start_beta = 1e-4
#β_T=0.02
end_beta = 0.02
steps = 25
n = 1024
#25要素の1次元テンソルβを生成
beta = torch.FloatTensor(steps)
#25要素の1次元テンソルαを生成
alpha = torch.FloatTensor(steps)
#25要素の1次元テンソルα_を生成
alpha_ = torch.FloatTensor(steps)

#βとαの値を計算する関数を定義
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
        #pandasのread_csvメソッドを使用して、pathからデータを読み込む
        df = pd.read_csv(path)
        #特徴量をarmdef.arm.spring_joint_count*2の大きさのリストとして、設定
        features = ([str(i) for i in range(armdef.arm.spring_joint_count*2)])
        #データフレームからxとyの値を抽出し、posに代入
        self.pos = df[['x', 'y']].values
        #データフレームから特徴量の値を抽出し、xに代入
        self.x = df[features].values
        #データフレームからθの値を抽出し、thetaに代入
        self.theta = df['theta'].values

    #xの長さを返す関数を定義
    def __len__(self):
        return len(self.x)
    #インデックスを受け取り、x、pos、thetaを返す関数を定義
    def __getitem__(self, idx):
        #インデックスを受け取り、xの値を浮動小数点数に変換
        x = torch.FloatTensor(self.x[idx])
        #インデックスを受け取り、posの値を浮動小数点数に変換
        pos = torch.FloatTensor(self.pos[idx])
        #インデックスを受け取り、thetaの値を浮動小数点数に変換
        theta = torch.FloatTensor([self.theta[idx]])
        return x, pos, theta


class PositionalEncoding(torch.nn.Module):
    def __init__(self, steps, d):
        #torch.nn.Moduleの__init__メソッドを呼び出す
        super().__init__()
        #step間隔で等間隔に配置された位置情報を生成し、1次元目に追加
        pos = torch.arange(steps).unsqueeze(1)
        #10000のd乗を計算し、divに代入
        #αは0からd-1までの偶数の値を持つテンソル
        div = torch.pow(10000, torch.arange(0, d, 2)/d)
        #要素がsteps×dのゼロテンソルを生成
        self.pe = torch.zeros(steps, d)
        #peの偶数列を, sin(pos/div)に設定
        self.pe[:, 0::2] = torch.sin(pos/div)
        #peの奇数列を, cos(pos/div)に設定
        self.pe[:, 1::2] = torch.cos(pos/div)
        #次元の設定
        self.d = d
    #順伝播処理を行う関数を定義
    def forward(self, x, step):
        #テンソルstepの形状をexpandメソッドを使用して、self.d×1の形状に変換し、転置
        step = step.expand(self.d, -1).T
        #テンソルpeの0次元に沿って値を抽出し、pe_に代入
        #stepテンソルをcpuに移動し、そのインデックスを使用して、self.peから値を抽出
        pe_ = torch.gather(self.pe, 0, step.cpu()).to(x.device)
        #テンソルxを√d倍して、pe_を加算
        x = x*math.sqrt(self.d)+pe_
        return x


class FC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.relu = nn.ReLU()
        #入力と出力の次元がdで線形結合を行う
        self.fc = nn.Linear(d, d)
        #dの次元に対してバッチ正規化を行う
        self.bn = nn.BatchNorm1d(d)
        #入力と出力の次元がdで線形結合を行う
        self.fc2 = nn.Linear(d, d)
    #順伝播処理を行う関数を定義
    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def extract(t, x_shape):
    #バッチサイズをテンソルtの形状の最初の次元のサイズとして設定
    batch_size = t.shape[0]
    #テンソルα_の最後の次元に沿って、tをcpuに移動し、そのインデックスを使用して、α_から値を抽出
    out = alpha_.gather(-1, t.cpu())
    #outの形状を変更する。
    #テンソルの最初の次元のサイズをバッチサイズに設定し、残りの次元を1に設定
    return out.reshape(batch_size, *((1,)*(len(x_shape) - 1))).to(t.device)

#拡散モデルの定義に基づいて、xtにノイズを加える
def gen_xt(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    #tをx0の形状に変換し、at_とする
    at_ = extract(t, x0.shape)
    #xを√at_×x0+√(1-at_)×noiseに設定
    x = torch.sqrt(at_)*x0+torch.sqrt(1-at_)*noise
    t = t.view(x.shape[0], 1)
    return x

#アームの出力(0~30)を正規化し、-1~1の範囲に正規化を行う関数を定義
def normalize(x: torch.Tensor):
    x -= 15
    x /= 15
    return x

#正規化処理をもとに戻す関数を定義
def denormalize(x: torch.Tensor):
    x *= 15
    x += 15
    return x
