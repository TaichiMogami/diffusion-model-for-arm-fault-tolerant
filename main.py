import os
import sys
import torch
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import pygame
import numpy as np
import tqdm
from simulator import definition as armdef
from diffusion_model import (
    ControlNet,
    ModelForXY,
    ModelForTheta,
    denoise_steps,
    extract,
    normalize,
    denormalize,
)


# ----------------------------------------------------------
# Main entry point
# ----------------------------------------------------------
def main():
    pygame.init()
    target_x, target_y = armdef.width / 2, armdef.height / 2 - 150
    model = load_model("data/controlnet_xy_and_theta.pth")
    #円状のパスを生成（各要素は[x,y]の座標）
    # target_path = draw_circle()
    # print(f"target path length: {len(target_path)}")
    # for pos in target_path:
    #     target_x, target_y = pos
    generate_control_signals(target_x, target_y, model)
    pygame.time.wait(50)
    pygame.quit()
    


def draw_circle():
    #円の軌道を描くための座標を格納するリストpathを設定
    path= []
    #円の中心のy座標をarmdef.width/2-100に設定
    y0 = armdef.height/2-100
    #円の中心のx座標をarmdef.height/2に設定
    x0 = armdef.width/2
    #円の半径を150に設定
    r = 150
    # 円の軌道を描くための座標を格納するリストlに座標を追加
    circle = np.arange(0,360,0.1)
    for i in tqdm.tqdm(range(len(circle))):
        x = x0 + r * np.cos(np.radians(i))
        y = y0 + r * np.sin(np.radians(i))
        path.append([x,y])
    return path
# ----------------------------------------------------------
# Load a model from a given path and put it on GPU
# ----------------------------------------------------------
def load_model(model_path):
    model = ControlNet(denoise_steps)
    model.load_state_dict(torch.load(model_path))
    return model.cuda()

#このファイルと同一のパス画像を保存する関数を定義
def save_image(display, filename):
    pygame.image.save(display, filename)
    

# ----------------------------------------------------------
# Generate control signals, visualize and plot data.
# This function builds target angles locally, calls controlnet
# to get the history of signals, applies filtering and plots data.
# ----------------------------------------------------------
def generate_control_signals(target_x, target_y, model):
    # Create the display window.
    display = pygame.display.set_mode((armdef.width, armdef.height))

    # Build the list of target theta values.
    target_thetas = [(3.14 / 2) * (i / 10) for i in tqdm.tqdm(range(-10, 10))]
    target_thetas = target_thetas + target_thetas[::-1]
    target_thetas *= 5
    print(f"target thetas length: {len(target_thetas)}")

    # Compute control signals for all target thetas.
    xt_all_runs = controlnet(target_x, target_y, denoise_steps, model, target_thetas)

    # Convert the control signals to a numpy array with float32 type.
    xt_all_runs_np = np.array(xt_all_runs, dtype=np.float32)

    # Reshape so that each row has 12 elements (assumed signal dimension).
    xt_all_runs_reshaped = xt_all_runs_np.reshape(-1, 12)

    # Create a dataframe from the reshaped array.
    df = pd.DataFrame(xt_all_runs_reshaped)

    # Apply a moving average filter.
    df_filtered = df

    # Plot both original and filtered data.
    # plot_data(df, df_filtered)
    
   # 既存の出力ファイルを取得
    existing_files = {f"output_frame_{i}.png" for i in range(15, 600, 40)}

    for i in range(df_filtered.shape[0]):
        if i in range(15, 600, 120):
            display.fill((255, 255, 255))  # 背景を白にリセット
            xt = df_filtered.iloc[i].to_list()
            print(f"xt: {xt}")

            draw_arm(target_x, target_y, xt, target_thetas[i], display, (0, 0, 255, int((i / 600) * 255)))
            save_image(display, f"output_frame_{i}.png")  # 画像を保存

    # # 不要なPNGファイルを削除
    # for file in os.listdir():
    #     if file.startswith("output_frame_") and file.endswith(".png"):
    #         if file not in existing_files:
    #             os.remove(file)
    #             print(f"Deleted: {file}")


# ----------------------------------------------------------
# Compute control signals for a series of target thetas.
# Returns a list of numpy arrays representing the history of xt.
# ----------------------------------------------------------
def controlnet(x, y, steps, model, target_thetas):
    # Initialize xt with a random tensor.
    xt = torch.randn(armdef.arm.spring_joint_count * 2).cuda()
    first = True
    current_steps = steps
    xt_history = []  # Stores xt at each target theta

    for theta in target_thetas:
        model.eval()
        pos = torch.FloatTensor([[x, y]]).cuda()
        theta_tensor = torch.FloatTensor([[theta]]).cuda()

        if not first:
            # Ensure xt is in float32 and normalized.
            xt = torch.tensor(xt, dtype=torch.float32).cuda()
            xt = normalize(xt)
            # Add noise and perform one denoising step.
            noise = torch.randn_like(xt).cuda()
            t = torch.tensor([current_steps - 1], dtype=torch.long).cuda()
            at_ = extract(t, xt.shape)
            xt_noised = torch.sqrt(at_) * xt + torch.sqrt(1 - at_) * noise
            xt = model.denoise(xt_noised, current_steps, pos, theta_tensor)
            xt = denormalize(xt)
        else:
            # For the first theta, use a different number of steps.
            current_steps = 4
            first = False

        xt_history.append(xt)

    # Convert each xt in the history to a numpy array.
    xt_history_np = [entry.cpu().detach().numpy() for entry in xt_history]
    return xt_history_np


# ----------------------------------------------------------
# Plot the original and filtered data side-by-side.
# Original data is plotted in red and filtered data in blue.
# ----------------------------------------------------------
def plot_data(df, df_filtered):
    fig, axes = plt.subplots(6, 2, figsize=(10, 15))
    for i in range(12):
        ax = axes[i % 6, i // 6]
        df[i].plot(ax=ax, title=f"input {i + 1}", color="red")
        df_filtered[i].plot(ax=ax, title=f"filtered input {i + 1}", color="blue")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Draw the arm on the display based on the control signal xt.
# ----------------------------------------------------------
def draw_arm(x, y, xt, theta, display,color):
    armdef.arm.calc(xt)
    pygame.draw.line(
        display,
        (255, 0, 0),
        (x, y),
        (np.cos(np.pi / 2 - theta) * 70 + x, np.sin(np.pi / 2 - theta) * 70 + y),
        5,
    )
    armdef.arm.draw(display,color)
    # font = pygame.font.Font(None, 24)
    # text1 = font.render(
    #     f"result: {armdef.arm.last.x[1] / np.pi * 180} degree", True, (0, 0, 0)
    # )
    # text2 = font.render(f"target: {theta / np.pi * 180} degree", True, (0, 0, 0))
    # text3 = font.render(
    #     f"error: {abs(armdef.arm.last.x[1] - theta) / np.pi * 180} degree",
    #     True,
    #     (0, 0, 0),
    # )
    # display.blit(text1, (10, 10))
    # display.blit(text2, (10, 40))
    # display.blit(text3, (10, 70))
    pygame.draw.circle(display, (0, 0, 0), (int(x), int(y)), 10)
    pygame.display.update()
    pygame.time.wait(50)


# ----------------------------------------------------------
# Apply a moving-average filter along the rows of a DataFrame.
# ----------------------------------------------------------
def moving_average_filter(df, window_size=3):
    return df.rolling(window=window_size, min_periods=1, axis=0).mean()


# ----------------------------------------------------------
# Perform spline interpolation on the DataFrame columns.
# Each of the first 12 columns is interpolated over n_points.
# ----------------------------------------------------------
def spline_interpolation(df, n_points=100):
    df_interpolated = pd.DataFrame()
    for i in range(12):
        df_interpolated[i] = np.interp(
            np.linspace(0, 1, n_points),
            np.linspace(0, 1, df.shape[0]),
            df[i],
        )
    return df_interpolated


if __name__ == "__main__":
    # display = pygame.display.set_mode((armdef.width, armdef.height))
    # display.fill((255, 255, 255))
    # color = (0,0,255,100)
    # pygame.draw.line(display, color, (0,0), (500,200), 5)
    # color = (0,0,255,200)
    # pygame.draw.line(display, color, (0,0), (200,500), 5)
    # pygame.display.update()
    # time.sleep(4)
    main()
