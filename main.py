import pygame
from pygame.locals import *
import numpy as np
import torch
import copy
import time
import pandas as pd
import os

from simulator import definition as armdef
from diffusion_model import Model, steps, extract, normalize, denormalize


# pathの点を順番に移動する
# while_sleep_timeは点を移動する間sleepする時間
# 滑らかに移動をさせるために, 1ステップ前の入力信号に数ステップ分ノイズを加え, 数ステップ分デノイズしている
def move(path, while_sleep_time=0):
    #pygameの初期化
    pygame.init()
    #armdef.width(1280)×armdef.height(720)の大きさの画面を作る
    display = pygame.display.set_mode((armdef.width, armdef.height))
    #Modelクラスのインスタンス生成
    model = Model(steps)
    #torch.loadで指定されたパスからモデルのパラメータを読み込む
    #load_state_dictメソッドを利用して、これらのパラメータを現在のモデルインスタンスにロードする
    model.load_state_dict(torch.load("data/model.pth"))
    #cudaメソッドを使用して、モデルのすべてのパラメータをGPUに転送する
    model = model.cuda()
    position = []
    #torch.randnメソッドを使用して、標準正規分布に従うランダムな値を持つテンソルを生成する
    #armdef.arm.spring_joint_count*2の大きさのテンソルを生成
    xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()
    #最初の実行時のみTrue
    first = True
    #steps_にsteps(25)を代入
    steps_ = steps
    #pathの点を順番に移動する
    for (x, y) in path:
        #モデルを評価モードに切り替える
        model.eval()
        #Yにyを代入
        Y = y
        #Xにxを代入
        X = x
        #torch.FloatTensorメソッドを使用して、XとYを要素とする1×2の32ビット浮動小数点数を要素とするテンソルを生成
        pos = torch.FloatTensor([[X, Y]]).cuda()
        #armにarmdef.armをdeepcoopy(深いコピー)したものを代入
        arm_ = copy.deepcopy(armdef.arm)
        #2週目以降の処理
        if not first:
            #xtを正規化
            xt = normalize(xt)
            #torch.randn_likeメソッドを使用して、標準正規分布に従う、xtと同じサイズのテンソルを生成
            noise = torch.randn_like(xt).cuda()
            #torch.Tensorメソッドを使用して、tに(steps-1)を要素として持つtensorを生成
            #longメソッドを使用して、tをfloat型からlong型に変換
            t = torch.Tensor([steps_-1]).long().cuda()
            #extractメソッドを使用して、at_にtとxt.shapeを引数として渡したものを代入
            at_ = extract(t, xt.shape)
            #拡散モデルの定義に基づいて、xtにノイズを加える
            xt = torch.sqrt(at_)*xt+torch.sqrt(1-at_)*noise
        #xtに対してデノイズ処理を行う
        xt = model.denoise(xt, steps_, pos)
        #xの正規化処理をもとに戻す
        xt = denormalize(xt)
        #xtをnumpy配列からリストに変換
        armdef.arm.calc(xt.tolist())
        # print(xt)
        #画面を白色で塗りつぶす
        display.fill((255, 255, 255))
        #pathの点を順番に描画
        for (x, y) in path:
            #指定された点に黒色の点を描画
            display.set_at((int(x), int(y)), (0, 0, 0))
            position.append((int(x), int(y)))    
        #アームを描画
        armdef.arm.draw(display)
        #画面を更新
        pygame.display.update()
        #点を移動する間は、プログラムの実行を一時停止
        time.sleep(while_sleep_time)

        #初回の処理
        if first:
            #steps_に4を代入
            steps_ = 4
            #初回の処理が終了したので、firstをFalseにする
            first = False

    pygame.quit()



# 円の軌道を描かせる
if __name__ == '__main__':
    #円の軌道を描くための座標を格納するリストlを設定
    l = []
    #円の中心のx座標をarmdef.height/2-100に設定
    y0 = armdef.height/2-100
    #円の中心のy座標をarmdef.width/2-100に設定
    x0 = armdef.width/2-100
    #円の半径を150に設定
    r = 150
    # #外れ値を除いた移動平均を計算する関数を定義
    # def filtered_mean(data, window_size=10):
    #     # データの平均を計算
    #     mean_x = sum([x for x, y in data])/len(data)
    #     mean_y = sum([y for x, y in data])/len(data)
    #     # 外れ値の基準を設定
    #     threshold = 2
    #     # 外れ値を除いたデータを格納するリストを設定
    #     filtered_data = [(x, y) for x, y in data if abs(x-mean_x) < threshold and abs(y-mean_y) < threshold]
    #     if len(filtered_data) == 0:
    #         return mean_x, mean_y
    #     filtered_mean_x = sum([x for x, y in filtered_data])/len(filtered_data)
    #     filtered_mean_y = sum([y for x, y in filtered_data])/len(filtered_data)
    #     return filtered_mean_x, filtered_mean_y
    
    #移動平均を計算する関数を定義
    def moving_average_2d(points, window_size=50):
        df = pd.DataFrame(points)
        df_ma = df.rolling(window=window_size, min_periods=1).mean()
        result = df_ma.values.tolist()
        return [tuple(points) for points in result]
    
    #3周分の円の軌道を描画
    for i in range(360*3):
        #iを度数法からラジアンに変換
        rad = np.deg2rad(i)
        #円の軌道を描くための座標をリストlに格納
        l += [(x0+r*np.cos(rad), y0+r*np.sin(rad))]
    #lの移動平均を計算
    low_pas_filtering = moving_average_2d(l)
    # print(f'low_pas_filtering[:10]:',low_pas_filtering[:10])
    # print(f'l[:10]:',l[:10])
    #l及びlow_pas_filteringの全てのデータをcsvファイルに書き込む
    def write_csv(data, filename):
        if not os.path.exists('l_data'):
            os.mkdir('l_data')
        df = pd.DataFrame(data, columns=['x', 'y'])
        df.to_csv(filename, index=False)
    
    write_csv(l, 'l_data/l.csv')
    write_csv(low_pas_filtering, 'l_data/low_pas_filtering.csv')    
    # # print(low_pas_filtering)
    # segment = l[100:107]
    # avg_x = sum([x for x, y in segment])/len(segment)
    # avg_y = sum([y for x, y in segment])/len(segment)
    # # print(sum(l[100:107])/7)
    # avg_point = (avg_x, avg_y)
    # print(avg_point)
    # print(low_pas_filtering[104])
    # print(len(low_pas_filtering))
    #円の軌道を0.001秒間隔で描画
    # move(l, 0.001)
    move(low_pas_filtering,  0.001)
    
    #low_pas_filtering上の点を線で結んだ上で描画
    
        
        

    # pygame.init()
    # display = pygame.display.set_mode((armdef.width, armdef.height))

    # model = Model(steps).cuda()
    # model.load_state_dict(torch.load("data/model.pth"))
    # model.eval()
    # for i in range(10):
    #     xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()
    #     pos = torch.FloatTensor([[150, 150]]).cuda()
    #     xt = model.denoise(xt,25,pos)
    #     xt = denormalize(xt)
    #     print(xt)
    #     armdef.arm.calc(xt.tolist())
        
    #     display.fill((255, 255, 255))
    #     armdef.arm.draw(display)

    #     pygame.display.update()
    #     time.sleep(1)