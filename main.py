import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import torch
from pygame.locals import *

from diffusion_model import Model, denormalize, extract, normalize, steps
from simulator import definition as armdef


def main():
    draw_cirtcle()


def move(path, while_sleep_time=0):
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))
    model = Model(steps).cuda()
    model.load_state_dict(torch.load("data/model.pth"))
    xt = torch.randn(armdef.arm.spring_joint_count * 2).cuda()
    first = True
    steps_ = steps
    xt_list = []

    # 全ての xt を保存
    for x, y in path:
        model.eval()
        pos = torch.FloatTensor([x, y]).cuda()

        if not first:
            xt_normalize = normalize(xt)
            noise = torch.randn_like(xt_normalize).cuda()
            t = torch.FloatTensor([steps_]).long().cuda()
            at_ = extract(t, xt.shape)
            xt_noised = torch.sqrt(at_) * xt_normalize + torch.sqrt(1 - at_) * noise
            xt = model.denoise(xt_noised, steps_, pos)
        else:
            xt = model.denoise(xt, steps_, pos)
            steps_ = 4
            first = False

        xt_list.append(denormalize(xt).cpu().detach().numpy())

    # 移動平均を適用
    xt_array = np.array(xt_list)
    df = pd.DataFrame(xt_array)
    smoothed_df = moving_average_filter(df, window_size=10)
    smoothed_xt_list = smoothed_df.values.tolist()
    # xt_list の計算を行ってその後、ローパスフィルタを適用
    for smoothed_xt in smoothed_xt_list:
        armdef.arm.calc(smoothed_xt)
        display.fill((255, 255, 255))
        for px, py in path:
            display.set_at((int(px), int(py)), (0, 0, 0))
        armdef.arm.draw(display)
        pygame.display.update()
        time.sleep(while_sleep_time)
    # プロットを実行
    plot_data(df, smoothed_df)
    pygame.quit()


# 円の軌道を描かせる
def draw_cirtcle():
    # 円の軌道を描くための座標を格納するリストlを設定
    path_coords = []
    # 円の中心のx座標をarmdef.height/2-100に設定
    y0 = armdef.height / 2 - 100
    # 円の中心のy座標をarmdef.width/2に設定
    x0 = armdef.width / 2
    # 円の半径を150に設定
    r = 150
    # 円の軌道を描くための座標を格納するリストlに座標を追加
    circle = np.arange(0, 360, 0.1)
    circle = np.round(circle, 1)
    for i in range(len(circle)):
        x = r * np.cos(np.radians(i)) + x0
        y = r * np.sin(np.radians(i)) + y0
        path_coords.append((x, y))
    move(path_coords, while_sleep_time=0.001)


# データのプロットを行う関数を定義
def plot_data(
    original_df,
    smoothed_df,
    num_cols=2,
    figure_size=(15, 10),
    title="Comparison of the Original and Smoothed Data",
):
    # 信号の数
    num_signals = len(original_df.columns)
    # 行数の計算
    num_rows = (num_signals + num_cols - 1) // num_cols
    # FigureとAxesの作成
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size)
    # np.ravel()で1次元配列に変換
    axes = np.ravel(axes)
    for i in range(num_signals):
        ax = axes[i]
        # x軸
        x_original = original_df.index.to_numpy()
        x_smoothed = smoothed_df.index.to_numpy()
        # y軸
        y_original = original_df.iloc[:, i].to_numpy()
        y_smoothed = smoothed_df.iloc[:, i].to_numpy()
        # 元データのプロット
        ax.plot(x_original, y_original, label=f"Original Data{i + 1}", color="blue")
        # 平滑化データのプロット
        ax.plot(x_smoothed, y_smoothed, label=f"Smoothed Data{i + 1}", color="red")
        # ラベルの設定
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal Value")
        ax.set_title(f"Signal{i + 1}")
        ax.legend()
    # 全体のタイトルの設定及びレイアウトの調整
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def moving_average_filter(df, window_size=10):
    return df.rolling(window=window_size, min_periods=1).mean()


# 平滑化処理を行う関数を定義
def apply_smoothing(target_pos, alpha=0.001):
    xt_transpose = torch.stack(target_pos).transpose(0, 1).cpu().detach()

    def smoothing(one_axis_list):
        smoothed = [one_axis_list[0]]
        for i in range(1, len(one_axis_list)):
            smoothed.append(one_axis_list[i] * alpha + smoothed[i - 1] * (1 - alpha))
        return smoothed

    smoothed_axes = [smoothing(one_axis) for one_axis in xt_transpose.numpy()]
    return np.array(smoothed_axes).T.tolist()


# low_pass_filteredを入力信号として、手先位置を計算する関数を定義
def calculate_end_effector_position(low_pass_filtered):
    # ロボットアームの手先位置を格納するリストを設定
    end_effector_positions = []
    # low_pass_filteredの要素をタプルからリストに変換
    low_pass_filtered = [list(xt) for xt in low_pass_filtered]
    # ロボットアームの手先位置を計算
    print("low_pass_filtered:", low_pass_filtered)
    for xt in low_pass_filtered:
        armdef.arm.calc(xt)
        end_effector_positions.append(copy.deepcopy(armdef.arm.end_effector))
    print("end_effector_positions:", end_effector_positions)
    return end_effector_positions


def write_csv(data, filename):
    if not os.path.exists("filtered_data"):
        os.mkdir("filtered_data")
    df = pd.DataFrame(data, columns=["x", "y"])
    df.to_csv(filename, index=False)


# 円の軌道を描かせる
if __name__ == "__main__":
    main()
