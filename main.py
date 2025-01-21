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
            target_pos.append((int(x), int(y)))
        # print(target_pos)   
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
    
    
    # target_posを初期化
    target_pos = []

    #円の軌道を描くための座標を格納するリストlに座標を追加
    for i in range(0, 360*3):
        x = r*np.cos(np.radians(i))+x0
        y = r*np.sin(np.radians(i))+y0
        l.append((x, y))
    
    #平滑化処理を行う関数を定義
    def apply_smoothing(target_pos, alpha=0.9):
        #平滑化の適用
        for i in range(1, len(target_pos)):
            pre_x, pre_y = l[-1]
            cur_x, cur_y = target_pos[i]
            #平滑化計算
            smoothed_x = alpha*pre_x+(1-alpha)*cur_x
            smoothed_y = alpha*pre_y+(1-alpha)*cur_y
            l.append((smoothed_x, smoothed_y))
        return l
    smoothed_path = apply_smoothing(target_pos)
    #l及びlow_pas_filteringの全てのデータをcsvファイルに書き込む
    def write_csv(data, filename):
        if not os.path.exists('l_data'):
            os.mkdir('l_data')
        df = pd.DataFrame(data, columns=['x', 'y'])
        df.to_csv(filename, index=False)
    
    write_csv(smoothed_path, 'l_data/l.csv')
    print(smoothed_path)
    move(smoothed_path,  0.001)
    
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