import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
import torch
import tqdm

from diffusion_model import Model, denormalize, extract, normalize, steps
from faults import FaultInjector
from simulator import definition as armdef


def main():
    draw_cirtcle()


def move(path, while_sleep_time=0, fault_type=None, fault_param=None):
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))
    model = Model(steps).cuda()
    model.load_state_dict(torch.load("data/model.pth"))
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)  # 複数GPUがある場合
    xt = torch.randn(armdef.arm.spring_joint_count * 2).cuda()
    first = True
    steps_ = steps
    xt_list = []
    pos_list = []
    end_effector_positions = []
    # 故障に関するインスタンスを生成
    fault_injector = FaultInjector(device="cuda")
    xt_buffer = []
    # 全ての xt を保存
    for step, (x, y) in enumerate(path):
        # 推論の実行
        model.eval()
        pos = torch.FloatTensor([x, y]).cuda()

        # xt = fault_injector.apply_lock_fault(
        #     xt,
        #     index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        #     lock_values=[0, 30, 30, 0, 0, 30, 30, 0, 0, 30, 30, 0],
        # )
        # xt = fault_injector.apply_noise_fault(xt, index=[4, 5], std_dev=10.0)
        # xt = fault_injector.apply_dropout_fault(xt, index=[4, 5], dropout_prob=1.0)
        xt = fault_injector.apply_delay_fault(xt, index=[4, 5], delay_steps=1200)

        if not first:
            xt_normalize = normalize(xt)
            noise = torch.randn_like(xt_normalize).cuda()
            t = torch.FloatTensor([steps_]).long().cuda()
            at_ = extract(t, xt.shape)
            xt_noised = torch.sqrt(at_) * xt_normalize + torch.sqrt(1 - at_) * noise
            xt = model.denoise(xt_noised, steps_, pos)
            # print("after denoise:", xt)
        else:
            # xt = torch.randn(armdef.arm.spring_joint_count * 2).cuda()
            xt = model.denoise(xt, steps_, pos)
            steps_ = 8
            first = False
        # prev_xt = xt.clone()
        xt_list.append(denormalize(xt).cpu().detach().numpy())
        pos_list.append(pos.cpu().detach().numpy())
    # pos_listの各要素にマイナスをつけて反転
    pos_list = [[-x, -y] for x, y in pos_list]
    # print(f"pos_list: {pos_list}")
    df_target_pos = pd.DataFrame(pos_list, columns=["target_x", "target_y"])
    print(f"df_target_pos: {df_target_pos}")
    # print(f"df_target_pos_shape: {df_target_pos.shape}")
    # print(f"df_pos_shape: {df_target_pos.shape}")
    # 移動平均を適用
    xt_array = np.array(xt_list)
    df = pd.DataFrame(xt_array)
    # print(f"df:\n{df}")
    smoothed_df = moving_average_filter(df, window_size=20)
    smoothed_xt_list = smoothed_df.values.tolist()
    # プロットを実行
    # xt_list の計算を行ってその後、ローパスフィルタを適用

    directory = "output_data"
    os.makedirs(directory, exist_ok=True)

    for index, smoothed_xt in enumerate(smoothed_xt_list):
        armdef.arm.calc(smoothed_xt)
        end_effector = -armdef.arm.last.x[0][0], -armdef.arm.last.x[0][1]
        # print(f"smoothed_xt: {smoothed_xt}, end_effector: {end_effector}")
        end_effector_positions.append(end_effector)
        display.fill((255, 255, 255))
        pygame.draw.lines(display, (0, 0, 0), False, path, 10)
        armdef.arm.draw(display)
        pygame.display.update()
        # surfaceをnumpy配列に変形
        # frame = pygame.surfarray.array3d(pygame.display.get_surface())
        # frameを転置
        # frame = np.transpose(frame, (1, 0, 2))
        # frames.append(frame)
        time.sleep(0.01)
    # clip = mpy.ImageSequenceClip(frames, fps=30)
    # clip.write_videofile("output.mp4", codec="libx264")
    # n output.mp4をoutput_dataディレクトリに保存
    # clip.write_videofile(os.path.join(directory, "output.mp4"), codec="libx264")
    # print(f"smoothed_df:\n{smoothed_df}")
    df_end_effector = pd.DataFrame(end_effector_positions, columns=["end_x", "end_y"])
    # print(f"df_end_effector:\n{df_end_effector}")
    pd.concat([df_target_pos, df_end_effector], axis=1).to_csv(
        os.path.join(directory, "target_and_end_effector.csv"), index=False
    )
    # calculate_distance(df_end_effector, df_target_pos)
    plot_target_and_end_effector(df_target_pos, df_end_effector)
    pygame.quit()


# def save_image(display, filename):
#     pygame.image.save(display, filename)


def plot_target_and_end_effector(df_target_pos, df_end_effector):
    import pandas as pd

    # --- ① Seriesだった場合にDataFrame化する ---
    if isinstance(df_target_pos, pd.Series):
        df_target_pos = df_target_pos.to_frame().T  # 列に変換し行に整形
    if isinstance(df_end_effector, pd.Series):
        df_end_effector = df_end_effector.to_frame().T

    # --- ② 必要な列を明示的に抽出（xとy） ---
    df_target_pos = df_target_pos[["target_x", "target_y"]]
    df_end_effector = df_end_effector[["end_x", "end_y"]]

    # --- ③ 中心化処理 ---
    target_pos_center = df_target_pos - df_target_pos.mean() + 200
    end_effector_center = df_end_effector - df_end_effector.mean() + 200

    # print(f"target_pos_center:\n{target_pos_center}")
    # print(f"end_effector_center:\n{end_effector_center}")
    # target_pos_centerとend_effector_centerとの距離を計算
    calculate_distance = np.sqrt(
        (target_pos_center["target_x"] - end_effector_center["end_x"]) ** 2
        + (target_pos_center["target_y"] - end_effector_center["end_y"]) ** 2
    )
    df_distance = pd.DataFrame(calculate_distance, columns=["Distance"])
    # mean_distance = df_distance.mean().item()
    print(f"average_df_distance:\n{df_distance.mean()}")
    # CSVファイルに保存
    if not os.path.exists("output_data"):
        os.mkdir("output_data")
    df_distance.to_csv("output_data/distance.csv", index=False)
    # df_distanceのプロット
    df_distance_index_ndarray = df_distance.index.to_numpy()
    df_distance_ndarray = df_distance.values.flatten()
    # print(f"df_distance_index_ndarray:\n{df_distance_index_ndarray}")
    # print(f"df_distance_ndarray:\n{df_distance_ndarray}")
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_distance_index_ndarray,
        df_distance_ndarray,
        label="Distance",
        color="green",
    )
    # print(f"target_pos_center:\n{target_pos_center}")
    # print(f"end_effector_center:\n{end_effector_center}")

    # --- ④ プロット処理 ---

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(
        target_pos_center["target_x"].values,
        target_pos_center["target_y"].values,
        label="Target Position",
        color="blue",
    )
    ax.plot(
        end_effector_center["end_x"].values,
        end_effector_center["end_y"].values,
        label="End Effector Position",
        color="red",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(3)

    ax.grid(color="gray", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.tick_params(labelsize=30)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=30)
    ax.set_ylabel("Y", fontsize=30)
    plt.show()
    # グラフの保存
    if not os.path.exists("output_data"):
        os.mkdir("output_data")
    fig.savefig("output_data/target_and_end_effector.png", dpi=300, bbox_inches="tight")


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
    circle = np.arange(0, 1800, 1)
    for i in tqdm.tqdm(range(len(circle))):
        x = r * np.cos(np.radians(i)) + x0
        y = r * np.sin(np.radians(i)) + y0
        path_coords.append((x, y))
    # print("path_coords:", path_coords)
    move(path_coords, while_sleep_time=0.001)


# xtの時系列変化をプロットする関数を定義
# smoothed_xt_listのデータをプロットする
# smoothed_dfのデータをsmoothed_dfのインデックスごとに12個のサブプロットに分けてプロットする
# 縦6個，横2個のサブプロットを作成し、各サブプロットにxtの時系列変化をプロットする
def plot_data(df, smoothed_df):
    fig, axs = plt.subplots(6, 2, figsize=(6, 12))
    for i in range(12):
        ax = axs[i // 2, i % 2]
        ax.plot(df.index, df[i], label=f"Original {i}", color="blue", alpha=0.5)
        ax.plot(smoothed_df.index, smoothed_df[i], label=f"Smoothed {i}", color="red")
        ax.set_title(f"Joint {i + 1}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.show()


def moving_average_filter(df, window_size=10):
    return df.rolling(window=window_size, min_periods=1).mean()


# # move関数内の引数で呼び出されているpathの座標とendeffectrorの座標の距離を計算する関数を定義
# def calculate_distance(df_end_effector, df_target_pos):
#     distances = []
#     df_end_effector_coords = list(zip(df_end_effector["x"], df_end_effector["y"]))
#     target_coords = list(zip(df_target_pos["x"], df_target_pos["y"]))
#     for (x, y), (target_x, target_y) in zip(df_end_effector_coords, target_coords):
#         distance = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
#         distances.append(distance)
#     average_distance = np.mean(distances)
#     # CSVファイルに保存
#     if not os.path.exists("output_data"):
#         os.mkdir("output_data")
#     df_distances = pd.DataFrame(distances, columns=["Distance"])
#     df_distances.to_csv("output_data/distances.csv", index=False)
#     # 平均距離を返す
#     print(f"Average distance: {average_distance:.2f}")
#     plt.plot(distances, label="Distance")
#     plt.xlabel("Time Step")
#     plt.ylabel("Distance")
#     plt.title("Distance between Path and End Effector")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     return average_distance


# # 平滑化処理を行う関数を定義
# def apply_smoothing(target_pos, alpha=0.001):
#     xt_transpose = torch.stack(target_pos).transpose(0, 1).cpu().detach()

#     def smoothing(one_axis_list):
#         smoothed = [one_axis_list[0]]
#         for i in range(1, len(one_axis_list)):
#             smoothed.append(one_axis_list[i] * alpha + smoothed[i - 1] * (1 - alpha))
#         return smoothed

#     smoothed_axes = [smoothing(one_axis) for one_axis in xt_transpose.numpy()]
#     return np.array(smoothed_axes).T.tolist()


# # low_pass_filteredを入力信号として、手先位置を計算する関数を定義
# def calculate_end_effector_position(low_pass_filtered):
#     # ロボットアームの手先位置を格納するリストを設定
#     end_effector_positions = []
#     # low_pass_filteredの要素をタプルからリストに変換
#     low_pass_filtered = [list(xt) for xt in low_pass_filtered]
#     # ロボットアームの手先位置を計算
#     print("low_pass_filtered:", low_pass_filtered)
#     for xt in low_pass_filtered:
#         armdef.arm.calc(xt)
#         end_effector_positions.append(copy.deepcopy(armdef.arm.end_effector))
#     print("end_effector_positions:", end_effector_positions)
#     return end_effector_positions


def write_csv(data, filename):
    if not os.path.exists("filtered_data"):
        os.mkdir("filtered_data")
    df = pd.DataFrame(data, columns=["x", "y"])
    df.to_csv(filename, index=False)


# 円の軌道を描かせる
if __name__ == "__main__":
    main()
