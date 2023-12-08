from enum import Enum
import math
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Optional


class SynapticType(Enum):
    PRE = 1
    POST = 2

class Synapse(object):
    def __init__(self, coordinate: list[float], synaptic_type: SynapticType):
        self.coordinate = coordinate
        self.synaptic_type = synaptic_type

    def move(self, det_x: float, det_y: float) -> None:
        self.coordinate[0] += det_x
        self.coordinate[1] += det_y

class NeuronCore:
    def __init__(self, coordinate: list[float], synapses: list[Synapse]):
        self.coordinate = coordinate
        if synapses is None:
            self.synapses = []
        else:
            self.synapses = synapses

    def draw_on_axes(self, pyplot_ax: matplotlib.axes.Axes) -> None:
        # Get all synapse coordinate.
        presynapse_x = []
        presynapse_y = []
        postsynapse_x = []
        postsynapse_y = []
        for synapse in self.synapses:
            if synapse.synaptic_type == SynapticType.PRE:
                presynapse_x.append(synapse.coordinate[0])
                presynapse_y.append(synapse.coordinate[1])
            elif synapse.synaptic_type == SynapticType.POST:
                postsynapse_x.append(synapse.coordinate[0])
                postsynapse_y.append(synapse.coordinate[1])
            else:
                raise Exception("Unknown synaptic type.")

        # Draw neuron core and synapses.
        pyplot_ax.scatter(x=self.coordinate[0], y=self.coordinate[1], s=5, c='red', marker='o')
        pyplot_ax.scatter(x=presynapse_x, y=presynapse_y, s=2, c='blue', marker='.')
        pyplot_ax.scatter(x=postsynapse_x, y=postsynapse_y, s=2, c='green', marker='.')

        # Draw dendrites and axons.
        for synapse in self.synapses:
            pyplot_ax.plot(
                [self.coordinate[0], synapse.coordinate[0]], [self.coordinate[1], synapse.coordinate[1]],
                lw='1', c='orange', ls='-'
            )

    def move(self, det_x: float, det_y: float, move_synapses: bool) -> None:
        if move_synapses is None:
            move_synapses = True

        self.coordinate[0] += det_x
        self.coordinate[1] += det_y

        if move_synapses:
            for synapse in self.synapses:
                synapse.move(det_x, det_y)

class NeuronController:
    """
    这些神经元自身会移动，自然会相对移动突触。
    突触在垂直神经元切线方向衍生速度与当前神经元趋向有关。
    突触根部沿着神经元表面旋转速度较慢，基本上神经元很少旋转。
    """

    def __init__(self, neuron: NeuronCore, seed: int):
        self.neuron = neuron
        if seed is None:
            self.seed = random.randint(0, 0x7FFF_FFFF_FFFF_FFFF)
        else:
            self.seed = seed

    def __gradient(
            self,
            origin_coordinate: list[float],
            centre_coordinate: list[float]
    ) -> Optional[float, None]:
        if origin_coordinate[0] == centre_coordinate[0]:
            if origin_coordinate[1] == centre_coordinate[1]:
                # Coincidence.
                return None
            # Perpendicular.
            return float('inf')
        return (origin_coordinate[1] - centre_coordinate[1]) / (origin_coordinate[0] - centre_coordinate[0])

    def __normalize_radian(self, a: float) -> float:
        a %= math.radians(360.0)
        if a < 0:
            a += math.radians(360.0)
        return a

    def __normalize_det_radian(self, a: float) -> float:
        #a = self.__normalize_radian(a)
        if a > math.pi:
            return 2 * math.pi - a
        elif a < -math.pi:
            return 2 * math.pi + a
        return a

    def do_tick(self, det_tick: float, trend_power: float, trend_rad: float) -> None:
        # 如何简单优化：加速=合并遍历；减少内存=重复利用数组。

        if trend_power < 1.0:
            raise ValueError('Trend power must be equals or greater than 1.0.')

        # 正则化趋向弧度。
        #trend_rad %= math.pi * 2.0
        #if trend_rad < 0.0:
        #    trend_rad += math.pi * 2.0
        if trend_rad < 0.0 or math.pi * 2.0 <= trend_rad:
            raise ValueError('Trend rad must be between 0.0 and 2*pi (include 0.0, but without'
                             '30 2*pi).')

        # Don't move entire neuron, 因为存在粘滞力，所以细胞核的移动和突触关系不大。
        # Move neuron core only.
        self.neuron.move(random.uniform(-1.0, 1.0) * det_tick, random.uniform(-1.0, 1.0) * det_tick, move_synapses=False)

        n = len(self.neuron.synapses)
        center_point = self.neuron.coordinate

        # 计算各个树/轴突与水平线之间夹角的弧度。
        rad_table = np.zeros(n)
        for i in range(0, n):
            synapse = self.neuron.synapses[i]
            if synapse.coordinate[0] == center_point[0]:
                if synapse.coordinate[1] == center_point[1]:
                    rad_table[i] = None
                else:
                    rad_table[i] = math.pi / 2
            else:
                k = (synapse.coordinate[1] - center_point[1])/(synapse.coordinate[0] - center_point[0])
                rad_table[i] = math.atan(k)
                if synapse.coordinate[1] < center_point[1] or (synapse.coordinate[1] == center_point[1] and synapse.coordinate[0] < center_point[0]):
                    rad_table[i] += math.pi

        # 计算各个树/轴突之间的角度差。
        diff_rad_table = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i, n):
                if rad_table[i] is None or rad_table[j] is None:
                    diff_rad_table[i][j] = diff_rad_table[j][i] = None
                else:
                    diff_rad_table[i][j] = math.fabs(rad_table[i] - rad_table[j])
                    diff_rad_table[i][j] = self.__normalize_det_radian(diff_rad_table[i][j])
                    diff_rad_table[j][i] = -diff_rad_table[i][j] #因为后面用不到，所以这一步可以省略。

        # 计算旋转倍率
        rotate_rate_table = np.zeros(n)
        for i in range(0, n):
            if rad_table[i] is not None:
                sum = 0.0
                count = 0
                for j in range(0, n):
                    if diff_rad_table[i][j] is not None and diff_rad_table[i][j] != 0.0:
                        if diff_rad_table[i][j] > 0.0:
                            sum += math.pi - diff_rad_table[i][j]
                        else:
                            sum -= math.pi + diff_rad_table[i][j]
                        count += 1
                rotate_rate_table[i] = sum / math.pi / count #math.pi是最大绝对值。

                curve_min = 0.0
                curve_max = 2.0
                if rotate_rate_table[i] >= 0.0:
                    rotate_rate_table[i] = math.pow(curve_max - curve_min, (rotate_rate_table[i] - 0.5) * 2.0) + curve_min
                else:
                    rotate_rate_table[i] = -(math.pow(curve_max - curve_min, (rotate_rate_table[i] + 0.5) * -2.0) + curve_min)

        normal_length_range = [3.0, 24.0]

        # 计算树/轴突的长度。（之后可以用synapse.__l来缓存）
        length_table = np.zeros(n)
        for i in range(0, n):
            synapse = self.neuron.synapses[i]
            if synapse.coordinate[0] == center_point[0]:
                if synapse.coordinate[1] == center_point[1]:
                    length_table[i] = 0.0
                else:
                    length_table[i] = math.fabs(synapse.coordinate[1] - center_point[1])
            else:
                length_table[i] = math.fabs(synapse.coordinate[0] - center_point[0]) / math.cos(rad_table[i])

        # 先旋转。
        normal_det_rad = math.radians(0.01)
        det_rad_max = math.radians(30.0)
        for i in range(0, n):
            synapse = self.neuron.synapses[i]
            det_rad = random.uniform(0, normal_det_rad) * rotate_rate_table[i]
            if det_rad > det_rad_max:
                det_rad = det_rad_max
            elif det_rad < -det_rad_max:
                det_rad = -det_rad_max
            rad_table[i] += det_rad
            synapse.coordinate = [
                center_point[0] + math.cos(rad_table[i]) * length_table[i],
                center_point[1] + math.sin(rad_table[i]) * length_table[i]
            ]

        # 计算趋向不相似度。
        trend_diff_table = np.zeros(n)
        for i in range(0, n):
            trend_diff_table[i] = trend_rad - rad_table[i]
            trend_diff_table[i] = self.__normalize_det_radian(trend_diff_table[i]) / math.pi

        # 计算生长倍率
        growth_rate_table = np.zeros(n)
        for i in range(0, n):
            curve_min = 0.1
            curve_max = 2.5
            curve_a = 3
            if trend_diff_table[i] >= 0.0:
                growth_rate_table[i] = (
                    (curve_max - curve_min) / 2 * math.pow((trend_diff_table[i] - 0.5) * 2.0, curve_a) +
                    (curve_max - curve_min) / 2 + curve_min
                )
            else:
                growth_rate_table[i] = -(
                    (curve_max - curve_min) / 2 * math.pow((trend_diff_table[i] + 0.5) * -2.0, curve_a) +
                    (curve_max - curve_min) / 2 + curve_min
                )

        for i in range(0, n):
            corrected_length_range = [normal_length_range[0] * growth_rate_table[i], normal_length_range[1] * growth_rate_table[i]]
            写到这里了。
