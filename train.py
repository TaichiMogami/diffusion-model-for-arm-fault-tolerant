import os

import torch
import torch.nn as nn
from tqdm import tqdm

from diffusion_model import Dataset, Model, gen_xt, normalize, steps

# 学習をおこない, パラメータをdata/model.pthに保存する
if __name__ == "__main__":
    # デバイスをcudaに設定
    device = "cuda"
    # Datasetクラス
    dataset = Dataset("data/train.csv")
    # modelにModelクラスのインスタンスを生成し、deviceを指定してGPUに転送
    model = Model(steps).to(device)
    # バッチサイズを100に設定
    batch_size = 512
    # データローダーの初期化
    dataloader = torch.utils.data.DataLoader(
        # データセットを指定
        dataset,
        # バッチサイズを指定
        batch_size=batch_size,
        # 各エポックの開始時にデータをシャッフルする
        shuffle=True,
        # データの読み込みを並列化するためのプロセス数を指定
        num_workers=14,
        # データセットのサイズがバッチサイズで割り切れない場合、最後のバッチを破棄する
        drop_last=True,
        # データをピンメモリに配置することを指定
        pin_memory=True,
    )

    # エポック数を20に設定
    epochs = 1000
    # Adamオプティマイザを使用して、モデルのパラメータを最適化（学習率は0.01）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # torch.optim.lr_scheduler.LinearLRメソッドを使用して、学習率を線形に減少させるスケジューラを生成
    scheduler = torch.optim.lr_scheduler.LinearLR(
        # optimizerをAdam,学習率の初期値を0.01,学習率の最終値を0.001,20エポックにわたって学習率を線形に減少させる
        optimizer,
        start_factor=1,
        end_factor=0.1,
        total_iters=epochs,
    )
    # 平均二乗誤差損失関数を定義
    criterion = nn.MSELoss()

    # 学習を開始
    for epoch in tqdm(range(epochs)):
        # total_lossを0に初期化
        total_loss = 0
        # 各バッチを(x, pos, theta)という形式のタプルとして取得し、enumerateメソッドを使用してデータローダーからバッチを順番に取得し、各バッチに対してインデックスを付ける。
        for batch, (x, pos, theta) in tqdm(enumerate(dataloader)):
            # データをGPUに転送
            x, pos, theta = x.to(device), pos.to(device), theta.to(device)
            # xを正規化
            x = normalize(x)
            # torch.randintメソッドを使用して、1からsteps(25)までの値をランダムに生成し、バッチサイズ（100）行1列のテンソルを生成
            t = torch.randint(1, steps, (batch_size,), device=device).long()
            # torch.randn_likeメソッドを使用して、正規分布に従うランダムな値を持つxと同じサイズのテンソルを生成
            y = torch.randn_like(x).to(device)
            # gen_xtメソッド(ノイズを生成）を使用して、x, t, yを引数として渡し、xを生成
            x = gen_xt(x, t, y)
            # x及びyをデバイスに転送
            x, y = x.to(device), y.to(device)
            # モデルにx, t, posを引数として渡し、予測値を取得
            pred = model(x, t, pos)
            # 平均二乗誤差損失関数を使用して、予測値とyの損失を計算
            loss = criterion(pred, y)
            # 勾配を初期化
            optimizer.zero_grad()
            # 逆伝播を実行
            loss.backward()
            # 学習率と最適化手法に基づいて、パラメータを更新
            optimizer.step()
            # epoch全体の損失を、各バッチの損失を累積することによって計算
            total_loss += loss.item()
        # エポック数を出力
        # 1バッチの損失の平均を出力
        print(f"epoch:{epoch} {total_loss / len(dataloader)}")
        # scheduler.stepメソッドを呼び出し、学習率を更新
        scheduler.step()

    # dataフォルダが存在しない場合、dataフォルダを作成
    if not os.path.exists("data"):
        os.mkdir("data")
    # モデルのパラメータをdata/model.pthに保存
    torch.save(model.state_dict(), "data/model.pth")
