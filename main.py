import pygame
from pygame.locals import *
import numpy as np
import torch
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt

from simulator import definition as armdef
from diffusion_model import ControlNet, ModelForXY, ModelForTheta, steps, extract, normalize, denormalize

#制御信号を保存するリスト
xt_list = []
# pathの点を順番に移動する
# while_sleep_timeは点を移動する間sleepする時間
# 滑らかに移動をさせるために, 1ステップ前の入力信号に数ステップ分ノイズを加え, 数ステップ分デノイズしている
def move(path, while_sleep_time=0):
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))
    model = ControlNet(steps)
    model.load_state_dict(torch.load("data/controlnet.pth"))
    model = model.cuda()
    xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()
    first = True
    steps_ = steps
    for (x, y) in path:
        model.eval()
        Y = y
        X = x
        pos = torch.FloatTensor([[X, Y]]).cuda()
        arm_ = copy.deepcopy(armdef.arm)
        if not first:
            xt = normalize(xt)
            noise = torch.randn_like(xt).cuda()
            t = torch.Tensor([steps_-1]).long().cuda()
            at_ = extract(t, xt.shape)
            xt = torch.sqrt(at_)*xt+torch.sqrt(1-at_)*noise

        xt = model.denoise(xt, steps_, pos)

        xt = denormalize(xt)
        armdef.arm.calc(xt.tolist())
        display.fill((255, 255, 255))
        for (x, y) in path:
            display.set_at((int(x), int(y)), (0, 0, 0))
        armdef.arm.draw(display)
        pygame.display.update()
        time.sleep(while_sleep_time)

        if first:
            steps_ = 4
            first = False

    pygame.quit()

# 円の軌道を描かせる
def draw_circle():
    l = []
    y0 = armdef.height/2-100
    x0 = armdef.width/2-100
    r = 150
    for i in range(360*3):
        rad = np.deg2rad(i)
        l += [(x0+r*np.cos(rad), y0+r*np.sin(rad))]
    move(l,  0.001)

def xy_and_theta(x, y, theta, display):
    model_xy = ModelForXY(steps)
    model_xy.load_state_dict(torch.load("data/model_for_xy.pth"))
    model_xy = model_xy.cuda()
    model_xy.eval()

    model_theta = ModelForTheta(steps)
    model_theta.load_state_dict(torch.load("data/model_for_theta.pth"))
    model_theta = model_theta.cuda()
    model_theta.eval()

    xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()

    Y = y
    X = x
    #
    pos = torch.FloatTensor([[X, Y]]).cuda()
    theta = torch.FloatTensor([[theta]]).cuda()

    for i in reversed(range(1, steps)):
        xt=model_xy.denoise_once(xt, i, pos)
        xt=model_theta.denoise_once(xt, i, theta)

    arm_ = copy.deepcopy(armdef.arm)

    xt = denormalize(xt)
    armdef.arm.calc(xt.tolist())
    display.fill((255, 255, 255))
    armdef.arm.draw(display)
    font = pygame.font.Font(None,  24)
    text1 = font.render(str(f"x: {armdef.arm.last.x[1]}"), True, (0, 0, 0))
    text2 = font.render(str(f"theta: {theta}"), True, (0, 0, 0))
    display.blit(text1, (10,10))
    display.blit(text2, (10,40))
    pygame.draw.circle(display, (0, 0, 0), (int(x), int(y)), 10)
    pygame.display.update()
    pygame.time.wait(100)

def controlnet(x, y, theta, display):
    model = ControlNet(steps)
    model.load_state_dict(torch.load("data/controlnet_xy_and_theta.pth"))
    #model.load_state_dict(torch.load("data/controlnet_xy.pth"))
    model = model.cuda()
    model.eval()

    xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()

    Y = y
    X = x
    pos = torch.FloatTensor([[X, Y]]).cuda()
    theta = torch.FloatTensor([[theta]]).cuda()
    xt = model.denoise(xt, steps, pos, theta)
    xt_list.append(xt)

#アームを描画する関数を設定
def draw_arm(x, y, xt, display):
    xt = denormalize(torch.tensor(xt)).cuda()
    armdef.arm.calc(xt.tolist())
    display.fill((255, 255, 255))
    pygame.draw.line(display, (255, 0, 0), (x, y), (np.cos(np.pi/2-theta)*70+x, np.sin(np.pi/2-theta)*70+y), 5)
    armdef.arm.draw(display)
    font = pygame.font.Font(None,  24)
    text1 = font.render(str(f"result: {armdef.arm.last.x[1]/3.14*180} degree"), True, (0, 0, 0))
    text2 = font.render(str(f"target: {theta/3.14*180} degree"), True, (0, 0, 0))
    display.blit(text1, (10,10))
    display.blit(text2, (10,40))
    pygame.draw.circle(display, (0, 0, 0), (int(x), int(y)), 10)
    pygame.display.update()
    pygame.time.wait(100)

#移動平均を適用する関数を定義
def moving_average_filter(df, window_size=10):
    return df.rolling(window=window_size, min_periods=1, axis=0).mean()

#データのプロットを行う関数を定義
def plot_data(original_df, smoothed_df):
    fig, axes = plt.subplots(6, 2, figsize=(20, 10))
    for i in range(original_df.shape[1]):
        ax = axes[i // 2, i % 2]
        ax.plot(
            original_df.index,
            original_df.iloc[:, i],
            label=f"Input Signal {i + 1}",
            linestyle="dashed",
        )
        ax.plot(
            smoothed_df.index, smoothed_df.iloc[:, i], label=f"Smoothed Data {i + 1}"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Signal Value")
        ax.set_title(f"Data {i + 1}")
        ax.legend()
    plt.suptitle("Comparison of the Original and Smoothed Data")
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))

    target_x = armdef.width/2
    target_y = armdef.height/2-150
    print("target_x = " , target_x)
    print("target_y = ", target_y)
    
    theta_values = [3.14 / 2 * i / 20 for i in range(-20, 20)]
    # additional_values= [j /10 for j in range(-10, 10)]
    for theta in (theta_values):
        controlnet(target_x, target_y, theta, display)
    # print("xt_list = ", xt_list)
    xt_array = torch.stack(xt_list).cpu().detach().numpy()
    df = pd.DataFrame(xt_array)
    # print("df = ", df)
    smoothed_df = moving_average_filter(df)
    # print("smoothed_df = ", smoothed_df)
    smoothed_xt_array = smoothed_df.values.tolist()
    
    #ローパスフィルタを適用した後のxtを用いて手先位置を計算し描画
    for xt in smoothed_xt_array:
        draw_arm(target_x, target_y, xt, display)
    
    #移動平均をかける前後のデータをプロット
    plot_data(df, smoothed_df)
    
    pygame.quit()

    #for i in range(-20,20):
    #    xy_and_theta(armdef.width/2, armdef.height/2-200, i/20, display)
    #pygame.quit()

    #l = []
    #for i in range(360):
    #    rad = np.deg2rad(i)
    #    l += [(armdef.width/2+100*np.cos(rad), armdef.height/2+100*np.sin(rad))]
    #move(l, 0.001)