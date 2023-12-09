from enum import Enum
import math
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import PySide6
import random
import time
import traceback
from typing import Union


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
        # 应该给突和触相同的颜色，并且点的大小根据当前分辨率调整。

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
        pyplot_ax.scatter(x=presynapse_x, y=presynapse_y, s=3, c='blue', marker='.')
        pyplot_ax.scatter(x=postsynapse_x, y=postsynapse_y, s=3, c='green', marker='.')

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

    def __init__(self, neuron: NeuronCore, seed: Union[int, None]):
        self.neuron = neuron
        if seed is None:
            self.seed = random.randint(0, 0x7FFF_FFFF_FFFF_FFFF)
        else:
            self.seed = seed

    @staticmethod
    def __normal(mu: float, sigma: float, x) -> float:
        # if sigma == 0:
        #     return 0  # 理论上，返回+/-inf更合理。但是毕竟这里无意义，就给予一个最简单的返回值。
        return 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(-(x - mu) ** 2 / (2 * sigma**2))

    def __my_dynamic_curve(self, y_max: float, a: float, x: float) -> float:
        # if a == 0:
        #     return y_max
        if x <= 0:
            return y_max
        # a should equals or greater than 0.
        return (self.__my_dynamic_curve_sub_function(a=a, x=x) *
                y_max / self.__my_dynamic_curve_sub_function(a=a, x=0))

    @staticmethod
    def __my_dynamic_curve_sub_function(a: float, x: float) -> float:
        temp = 10.0 * a * (x - 2**(-a))
        try:
            return 2.0 / (1.0 + math.exp(temp))
        except OverflowError as err:
            print('Caught an OverflowError.')
            traceback.print_exc()
            if temp > 0:
                return 0
            elif temp < 0:
                return 2.0
            else:
                raise err

    @staticmethod
    def __gradient(
            origin_coordinate: list[float],
            centre_coordinate: list[float]
    ) -> Union[float, None]:
        if origin_coordinate[0] == centre_coordinate[0]:
            if origin_coordinate[1] == centre_coordinate[1]:
                # Coincidence.
                return None
            # Perpendicular.
            return float('inf')
        return (origin_coordinate[1] - centre_coordinate[1]) / (origin_coordinate[0] - centre_coordinate[0])

    @staticmethod
    def __normalize_radian(a: float) -> float:
        a %= math.radians(360.0)
        if a < 0:
            a += math.radians(360.0)
        return a

    @staticmethod
    def __normalize_det_radian(a: float) -> float:
        # a = self.__normalize_radian(a)
        if a > math.pi:
            return 2 * math.pi - a
        elif a < -math.pi:
            return 2 * math.pi + a
        return a

    def do_tick(self, det_tick: float, trend_power: Union[float, None], trend_rad: [float, None]) -> None:
        # 如何简单优化：加速=合并遍历；减少内存=重复利用数组。

        if trend_power is None:
            trend_power = 1.0
        elif trend_power < 1.0: # 需要：让trend_power在(0,1)上有效。
            raise ValueError('trend_power must be equals or greater than 1.0.')

        # 正则化趋向弧度。
        # trend_rad %= math.pi * 2.0
        # if trend_rad < 0.0:
        #     trend_rad += math.pi * 2.0
        if trend_rad < 0.0 or math.pi * 2.0 < trend_rad:
            raise ValueError('trend_rad must be between 0.0 and 2*pi (include 0.0, but without'
                             '30 2*pi).')

        normal_length_range = [30.0, 40.0]
        if normal_length_range[0] > normal_length_range[1]:
            raise ValueError('Incorrect range of normal_length_range.')
        if normal_length_range[0] < 0.0:
            raise ValueError('Incorrect range of normal_length_range.')
        normal_det_rad = math.radians(1.0)
        det_rad_max = math.radians(1.0)
        max_det_length = 12.0

        允许跨中心点移动，允许length为负。
        允许没有trend_rad，即不进行趋向运算。
        在trend_power==1.0时应让求昂算法无效，请改进运算来达到此目的。

        # Don't move entire neuron, 因为存在粘滞力，所以细胞核的移动和突触关系不大。
        # Move neuron core only.
        self.neuron.move(
            random.uniform(-1.0, 1.0) * det_tick,
            random.uniform(-1.0, 1.0) * det_tick,
            move_synapses=False
        )

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
                if (synapse.coordinate[1] < center_point[1] or
                   (synapse.coordinate[1] == center_point[1] and synapse.coordinate[0] < center_point[0])):
                    rad_table[i] += math.pi

        # 计算各个树/轴突之间的角度差。
        diff_rad_table = np.zeros((n, n))
        for i in range(0, n):
            for j in range(i, n):
                if rad_table[i] is None or rad_table[j] is None:
                    diff_rad_table[i][j] = diff_rad_table[j][i] = None
                else:
                    diff_rad_table[i][j] = math.fabs(rad_table[i] - rad_table[j])
                    diff_rad_table[i][j] = self.__normalize_det_radian(float(diff_rad_table[i][j]))
                    diff_rad_table[j][i] = -diff_rad_table[i][j]  # 因为后面用不到，所以这一步可以省略。

        # 计算旋转倍率
        rotate_rate_table = np.zeros(n)
        for i in range(0, n):
            if rad_table[i] is not None:
                diff_rad_sum = 0.0
                count = 0
                for j in range(0, n):
                    if diff_rad_table[i][j] is not None and diff_rad_table[i][j] != 0.0:
                        if diff_rad_table[i][j] > 0.0:
                            diff_rad_sum += math.pi - diff_rad_table[i][j]
                        else:
                            diff_rad_sum -= math.pi + diff_rad_table[i][j]
                        count += 1
                if count != 0:
                    rotate_rate_table[i] = diff_rad_sum / math.pi / count  # math.pi是最大绝对值。
                else:
                    rotate_rate_table[i] = 0.0

                curve_min = 0.0
                curve_max = 2.0
                if rotate_rate_table[i] >= 0.0:
                    rotate_rate_table[i] = (math.pow(curve_max - curve_min, (rotate_rate_table[i] - 0.5) * 2.0) +
                                            curve_min)
                else:
                    rotate_rate_table[i] = -(math.pow(curve_max - curve_min, (rotate_rate_table[i] + 0.5) * -2.0) +
                                             curve_min)

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
        for i in range(0, n):
            synapse = self.neuron.synapses[i]
            det_rad = random.uniform(0, normal_det_rad) * rotate_rate_table[i]
            if det_rad > det_rad_max:
                det_rad = det_rad_max
            elif det_rad < -det_rad_max:
                det_rad = -det_rad_max
            rad_table[i] = self.__normalize_radian(float(rad_table[i]) + det_rad)
            synapse.coordinate = [
                center_point[0] + math.cos(rad_table[i]) * length_table[i],
                center_point[1] + math.sin(rad_table[i]) * length_table[i]
            ]  # 这步是否多余？

        # 计算趋向不相似度。
        trend_diff_table = np.zeros(n)
        for i in range(0, n):
            trend_diff_table[i] = trend_rad - rad_table[i]
            trend_diff_table[i] = self.__normalize_det_radian(float(trend_diff_table[i])) / math.pi
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

        # 计算趋向后的生长数值
        det_length_table = np.zeros(n)
        if normal_length_range[0] != normal_length_range[1]:
            for i in range(0, n):
                # 计算趋向后的长度范围
                corrected_length_range = [
                    normal_length_range[0] * growth_rate_table[i],
                    normal_length_range[1] * growth_rate_table[i]
                ]
                range_length = corrected_length_range[1] - corrected_length_range[0]
                half_range_length = range_length / 2
                middle = (corrected_length_range[0] + corrected_length_range[1]) / 2

                # 计算生长差。
                diff_length = length_table[i] - middle
                reg_diff_length = diff_length / half_range_length

                det_l = random.uniform(-1.0, 1.0)
                # 计算生长偏离度。
                sigma = 1.0 / 3.0  # reg_diff_length_max = 1.0
                normal_fix_rate = 1.0 / self.__normal(0.0, sigma, 0.0)
                det_l_rate_max = self.__normal(0.0, sigma, reg_diff_length) * normal_fix_rate
                if reg_diff_length >= 0:
                    det_l *= self.__my_dynamic_curve(det_l_rate_max, float(reg_diff_length), det_l)
                else:
                    det_l *= self.__my_dynamic_curve(det_l_rate_max, -float(reg_diff_length), -det_l)

                if det_l > max_det_length:
                    det_l = max_det_length
                elif det_l < -max_det_length:
                    det_l = -max_det_length
                det_length_table[i] = det_l
        else:
            for i in range(0, n):
                # 计算趋向后的长度范围
                range_length = 0.0
                middle = normal_length_range[0] * growth_rate_table[i]

                # 计算生长差。
                diff_length = length_table[i] - middle
                reg_diff_length = diff_length
                # if reg_diff_length > 0.0:
                #     reg_diff_length += 1.0
                # elif reg_diff_length < 0.0:
                #     reg_diff_length -= 1.0

                det_l = random.uniform(-1.0, 1.0)
                # 计算生长偏离度。
                sigma = 1.0 / 3.0  # reg_diff_length_max = 1.0
                normal_fix_rate = 1.0 / self.__normal(0.0, sigma, 0.0)
                det_l_rate_max = self.__normal(0.0, sigma, reg_diff_length) * normal_fix_rate
                if diff_length >= 0:
                    det_l *= self.__my_dynamic_curve(det_l_rate_max, float(reg_diff_length), det_l)
                else:
                    det_l *= self.__my_dynamic_curve(det_l_rate_max, -float(reg_diff_length), -det_l)

                if det_l > max_det_length:
                    det_l = max_det_length
                elif det_l < -max_det_length:
                    det_l = -max_det_length
                det_length_table[i] = det_l


        # 生长
        for i in range(0, n):
            synapse = self.neuron.synapses[i]

            if length_table[i] == 0.0:
                rad_table[i] = random.uniform(0.0, 2.0 * math.pi)
            length_table[i] += det_length_table[i]
            if length_table[i] < 0.0:
                length_table[i] = 0.0
            synapse.coordinate = [
                center_point[0] + math.cos(rad_table[i]) * length_table[i],
                center_point[1] + math.sin(rad_table[i]) * length_table[i]
            ]


def demo() -> None:
    print('Lively Neuron Demo.')

    nn = 1
    sn_range = [20, 50]
    init_stage_size = [[0, 0], [100, 100]]

    random.seed(time.time())

    neurons = []
    neuron_controllers = []
    for i in range(0, nn):
        synapses = []
        sn = random.randint(sn_range[0], sn_range[1])
        for j in range(0, sn):
            synapse = Synapse(
                [random.uniform(init_stage_size[0][0], init_stage_size[1][0]),
                 random.uniform(init_stage_size[0][1], init_stage_size[1][1])],
                random.choice([SynapticType.PRE, SynapticType.POST])
            )
            synapses.append(synapse)
        neuron = NeuronCore(
            [random.uniform(init_stage_size[0][0], init_stage_size[1][0]),
             random.uniform(init_stage_size[0][1], init_stage_size[1][1])],
            synapses
        )
        neurons.append(neuron)
        neuron_controller = NeuronController(neuron, None)
        neuron_controllers.append(neuron_controller)

    import matplotlib
    matplotlib.use('QtAgg')

    plt.ion()

    while True:
        try:
            for neuron in neurons:
                neuron.draw_on_axes(plt)

            for neuron_controller in neuron_controllers:
                neuron_controller.do_tick(1.0, 1.0, random.uniform(0.0, math.pi * 2.0))

            plt.pause(0.01)
            plt.clf()
        except InterruptedError:
            break

    plt.ioff()


if __name__ == '__main__':
    demo()
