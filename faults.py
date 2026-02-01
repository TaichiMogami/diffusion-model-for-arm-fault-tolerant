import math
import random
from collections import deque

import torch


class FaultInjector:
    def __init__(self, device="cuda"):
        # ジョイントごとの遅延バッファーを保持する辞書を定義
        self.device = device
        self.delay_buffers = {}

    def apply_lock_fault(self, xt, index, lock_values):
        # xtの一部の入力を固定値に設定
        # xt:入力テンソル，index:固定するインデックス，lock_values:各インデックスに対応する固定値
        # 故障適用後のテンソルを返す．
        xt = xt.clone()
        for i, val in zip(index, lock_values):
            xt[i] = val
        return xt

    def apply_noise_fault(self, xt, index, std_dev=1.0):
        # xtの一部のインデックスにガウスノイズを付加
        # 故障適用後のテンソルを返す
        xt = xt.clone()
        for i in index:
            noise = torch.randn((), device=self.device) * std_dev
            xt[i] += noise
        return xt

    def apply_dropout_fault(self, xt, index, dropout_prob):
        # xtの指定したインデックスをdropout_probの確率でゼロにする.
        # 故障適用後のテンソルを返す
        xt = xt.clone()
        for i in index:
            if random.random() < dropout_prob:
                xt[i] = 0.0
        return xt

    def apply_delay_fault(self, xt, index, delay_steps=50):
        # xtを内部バッファーxt_delay_bufferに保持し，delay_steps分遅らせて返す
        # 遅延故障適用後のテンソルを返す
        for i in index:
            if i not in self.delay_buffers:
                self.delay_buffers[i] = []
            self.delay_buffers[i].append(xt[i].clone())

            if len(self.delay_buffers[i]) > delay_steps:
                delayed_value = self.delay_buffers[i].pop(0)
                xt[i] = delayed_value
        return xt

    def reset_buffer(self):
        # 遅延用バッファーのリセット
        self.xt_buffer = []
