import torch
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pygame
import numpy as np
import tqdm

from simulator import definition as armdef
from diffusion_model import (
    ControlNet,
    ModelForXY,
    ModelForTheta,
    steps,
    extract,
    normalize,
    denormalize,
)

# 制御信号を保存するリスト
xt_list = []
# thetaの値を保存するリスト
target_thetas = []
# 全てのxtの履歴を保存するリスト
xt_all_runs = []


def main():
    pygame.init()
    target_x, target_y = armdef.width / 2, armdef.height / 2 - 150
    model = load_model("data/controlnet_xy_and_theta.pth")
    generate_control_signals(target_x, target_y, model)
    pygame.quit()


def load_model(model_path):
    model = ControlNet(steps)
    model.load_state_dict(torch.load(model_path))
    return model.cuda()


def generate_control_signals(target_x, target_y, model):
    global xt_all_runs, target_thetas  # グローバル変数を参照
    display = pygame.display.set_mode((armdef.width, armdef.height))
    for i in tqdm.tqdm(range(-20, 20)):
        theta = (3.14 / 2) * (i / 20)
        target_thetas.append(theta)

        # 制御信号を生成し，保存
        xt_history = controlnet(target_x, target_y, theta, steps, model, display)
        xt_all_runs.append(xt_history)

    # xt_all_runsをnumpyのfloat32に変換
    xt_all_runs_np = np.array(xt_all_runs, dtype=np.float32)

    # 2次元の形状に変換
    xt_all_runs_reshaped = xt_all_runs_np.reshape(-1, 12)

    # データフレームに変換
    df = pd.DataFrame(xt_all_runs_reshaped)

    # 移動平均フィルタを適用
    df_filtered = moving_average_filter(df)

    # データをプロット
    plot_data(df, df_filtered)
    # 描画処理
    for run_idx, xt_history in enumerate(xt_all_runs):
        theta = target_thetas[run_idx]  # 対応するthetaを取得
        for xt in xt_history:
            draw_arm(armdef.width / 2, armdef.height / 2 - 150, xt, theta, display)
    return df_filtered, target_thetas


def controlnet(x, y, theta, steps, model, display):
    global xt_all_runs  # グローバル変数を参照
    xt = torch.randn(armdef.arm.spring_joint_count * 2).cuda()
    first = True
    steps_ = steps
    xt_history = []  # xtの履歴を保存するリスト

    for i in reversed(range(1, steps_)):
        model.eval()
        pos = torch.FloatTensor([[x, y]]).cuda()
        theta_tensor = torch.FloatTensor([[theta]]).cuda()

        if not first:
            xt = torch.tensor(xt, dtype=torch.float32).cuda()
            xt = normalize(xt)
            noise = torch.randn_like(xt).cuda()
            t = torch.FloatTensor([i]).long().cuda()
            at_ = extract(t, xt.shape)
            xt_noised = torch.sqrt(at_) * xt + torch.sqrt(1 - at_) * noise
            xt = model.denoise(xt_noised, i, pos, theta_tensor)
            xt = denormalize(xt)
            xt = xt.tolist()
        else:
            for _ in range(4):
                xt = torch.tensor(xt, dtype=torch.float32).cuda()
                xt = normalize(xt)
                noise = torch.randn_like(xt).cuda()
                t = torch.FloatTensor([i]).long().cuda()
                at_ = extract(t, xt.shape)
                xt_noised = torch.sqrt(at_) * xt + torch.sqrt(1 - at_) * noise
                xt = model.denoise(xt_noised, i, pos, theta_tensor)
                xt = denormalize(xt)
                xt = xt.tolist()
            first = False
    xt_history.append(xt)  # step数分のxtの履歴を保存
    return xt_history  # return each xt data


# dfとdf_filteredを引数に取り，プロットする関数を定義
def plot_data(df, df_filtered):
    num_signals = df.shape[1]
    num_columns = 2
    num_rows = (num_signals + num_columns - 1) // num_columns
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))
    axes = np.ravel(axes)

    for i in range(num_signals):
        ax = axes[i]
        plot_signal(ax, df, df_filtered, i)

    fig.suptitle("Original and filtered signals")
    plt.tight_layout()
    plt.show()


def plot_signal(ax, df, df_filtered, signal_index):
    x_df = df.index.to_numpy()
    x_filtered_df = df_filtered.index.to_numpy()
    y_df = df.iloc[:, signal_index].to_numpy()
    y_filtered_df = df_filtered.iloc[:, signal_index].to_numpy()

    ax.plot(x_df, y_df, label=f"Original signal {signal_index + 1}", color="blue")
    ax.plot(
        x_filtered_df,
        y_filtered_df,
        label=f"Filtered signal {signal_index + 1}",
        color="red",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal value")
    ax.set_title(f"Signal {signal_index + 1}")
    ax.legend()


def draw_arm(x, y, xt, theta, display):
    armdef.arm.calc(xt)
    display.fill((255, 255, 255))
    pygame.draw.line(
        display,
        (255, 0, 0),
        (x, y),
        (np.cos(np.pi / 2 - theta) * 70 + x, np.sin(np.pi / 2 - theta) * 70 + y),
        5,
    )
    armdef.arm.draw(display)
    font = pygame.font.Font(None, 24)
    text1 = font.render(
        f"result: {armdef.arm.last.x[1] / np.pi * 180} degree", True, (0, 0, 0)
    )
    text2 = font.render(f"target: {theta / np.pi * 180} degree", True, (0, 0, 0))
    text3 = font.render(
        f"error: {(armdef.arm.last.x[1] - theta) / np.pi * 180}", True, (0, 0, 0)
    )
    display.blit(text1, (10, 10))
    display.blit(text2, (10, 40))
    display.blit(text3, (10, 70))
    pygame.draw.circle(display, (0, 0, 0), (int(x), int(y)), 10)
    pygame.display.update()
    pygame.time.wait(100)


def moving_average_filter(df, window_size=3):
    return df.rolling(window=window_size, min_periods=1, axis=0).mean()


if __name__ == "__main__":
    main()
