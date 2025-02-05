import torch
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import pygame
import numpy as np

from simulator import definition as armdef
from diffusion_model import ControlNet, ModelForXY, ModelForTheta, steps, extract, normalize, denormalize

# 制御信号を保存するリスト
xt_list = []
# thetaの値を保存するリスト
target_thetas = []
#全てのxtの履歴を保存するリスト
xt_all_runs = []
def main():
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))
    target_x, target_y = armdef.width / 2, armdef.height / 2 - 150
    model = load_model("data/controlnet_xy_and_theta.pth")

    xt_list, target_thetas = generate_control_signals(target_x, target_y, model)
    smoothed_xt_list = smooth_control_signals(xt_list)
    for xt, theta in zip(smoothed_xt_list, target_thetas):
        draw_arm(target_x, target_y, xt, theta, display)
    plot_xt_evolution(xt_all_runs)

    pygame.quit()

def load_model(model_path):
    model = ControlNet(steps)
    model.load_state_dict(torch.load(model_path))
    return model.cuda()

def generate_control_signals(target_x, target_y, model):
    xt_list, target_thetas = [], []
    for i in range(-20, 20):
        theta = (3.14 / 2) * (i / 20)
        xt = controlnet(target_x, target_y, theta, steps, model,display=pygame.display.set_mode((armdef.width, armdef.height)))
        xt_list.append(xt)
        target_thetas.append(theta)
    return xt_list, target_thetas

def smooth_control_signals(xt_list):
    xt_array = torch.stack(xt_list).numpy()
    df = pd.DataFrame(xt_array)
    smoothed_df = df
    return smoothed_df.to_numpy().tolist()

def controlnet(x, y, theta, steps, model, display):
    xt = torch.randn(armdef.arm.spring_joint_count * 2).cuda()
    first = True
    steps_ = steps
    xt_history = [] # xtの履歴を保存するリスト
    
    for i in reversed(range(1, steps_ )):
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
        draw_arm(x, y, xt, theta, display) #デノイズのstepごとにアームを描画
        xt_history.append(xt) #step数分のxtの履歴を保存
        
    xt_all_runs.append(xt_history) # 40回分のxtの履歴を保存
    return xt  

#xtの各ステップの値を平均してプロットする関数を定義
def plot_xt_evolution(xt_history):
    xt_all_runs_np = np.array([np.array([xt.cpu().detach().numpy() for xt in run]) for run in xt_all_runs])  # (40, steps, 12)
    xt_mean = np.mean(xt_all_runs_np, axis=0)  # (steps, 12) - 40回分の平均をとる
    steps = np.arange(len(xt_mean))

    num_dims = xt_mean.shape[1]  # 12次元
    fig, axes = plt.subplots(num_dims, 1, figsize=(8, 2 * num_dims), sharex=True)

    for i in range(num_dims):
        axes[i].plot(steps[::-1], xt_mean[:, i], marker="o", linestyle="-", label=f"Dim {i+1}")  # 逆順に修正
        axes[i].set_ylabel(f"Dim {i+1}")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Denoising Step")  # 最後のプロットに x 軸ラベルを設定
    fig.suptitle("Mean Evolution of xt over denoising steps (averaged over 40 runs)")
    plt.tight_layout()
    plt.show()


def draw_arm(x, y, xt, theta, display):
    armdef.arm.calc(xt)
    display.fill((255, 255, 255))
    pygame.draw.line(display, (255, 0, 0), (x, y), (np.cos(np.pi / 2 - theta) * 70 + x, np.sin(np.pi / 2 - theta) * 70 + y), 5)
    armdef.arm.draw(display)
    font = pygame.font.Font(None, 24)
    text1 = font.render(f"result: {armdef.arm.last.x[1] / np.pi * 180} degree", True, (0, 0, 0))
    text2 = font.render(f"target: {theta/ np.pi * 180} degree", True, (0, 0, 0))
    display.blit(text1, (10, 10))
    display.blit(text2, (10, 40))
    pygame.draw.circle(display, (0, 0, 0), (int(x), int(y)), 10)
    pygame.display.update()
    pygame.time.wait(100)

def moving_average_filter(df, window_size=3):
    return df.rolling(window=window_size, min_periods=1, axis= 0).mean()


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

if __name__ == '__main__':
    main()