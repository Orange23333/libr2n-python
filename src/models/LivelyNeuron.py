from enum import Enum
import math
import matplotlib.axes
import matplotlib.pyplot as plt
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

    def __normalize_radian(self, rad: float) -> float:
        rad = rad % math.radians(360.0)
        if rad < 0:
            rad += math.radians(360.0)
        return rad

    def __k_to_rad(self, k: Optional[float, None]) -> Optional[float, None]:
        if k is None:
            return None
        return math.atan(k)

    def __expand_length(
            self,
            origin_coordinate: list[float],
            centre_coordinate: list[float],
            expand_length: float
    ) -> list[float]:
        k = self.__gradient(
            origin_coordinate=origin_coordinate,
            centre_coordinate=centre_coordinate
        )
        if k is None:
            return [0, 0]
        k_rad = math.atan(k)

        # Return delta coordinate.
        return [
            expand_length * math.cos(k_rad),
            expand_length * math.sin(k_rad)
        ]

    def __trend_angle(
            self,
            origin_coordinate: list[float],
            centre_coordinate: list[float],
            origin_det_length: float,
            origin_det_rad: float,
            trend_power: float,
            trend_angle_rad: float
    ) -> list[float]:
        重写！！！

        if trend_power < 1.0:
            raise ValueError('Trend power must be equals or greater than 1.0.')

        k = self.__gradient(
            origin_coordinate=origin_coordinate,
            centre_coordinate=centre_coordinate
        )
        if k is None:
            return [
                math.cos(trend_angle_rad) * trend_power,
                math.sin(trend_angle_rad) * trend_power
            ]
        k_rad = self.__normalize_radian(math.atan(k))
        trend_angle_rad = self.__normalize_radian(trend_angle_rad)

        # 实际上就是趋向单位向量在树/轴突上的投影的正则化。
        trend_similarity = (math.cos(self.__normalize_radian(k_rad - trend_angle_rad)) + 1.0) / 2.0
        # 以某种函数来处理正则化的数值为倍率，通常大于0，斜率在0.5附近突变。
        ts_result_min = 0.1
        trend_similarity = math.pow(trend_power + 1.0 - ts_result_min, trend_similarity) - (1.0 - ts_result_min)

        trend = [
            trend_similarity * math.cos(k_rad),
            trend_similarity * math.sin(k_rad)
        ]

        # Return delta coordinate.
        return [

        ]

    def do_tick(self, det_tick: float, trend_power: float, trend_angle_rad: float) -> None:
        # Move entire neuron.
        self.neuron.move(random.uniform(-1.0, 1.0) * det_tick, random.uniform(-1.0, 1.0) * det_tick, move_synapses=True)

        # Move each synapse.
        for synapse in self.neuron.synapses:
            det_l = random.uniform(-1.0, 1.0) * det_tick
            det_rad = math.radians(0.01)
            det_rad = random.uniform(-det_rad, det_rad)
            det_coordinate = self.__trend_angle(
                origin_coordinate=synapse.coordinate,
                centre_coordinate=self.neuron.coordinate,
                origin_det_length=det_l,
                origin_det_rad=det_rad,
                trend_power=trend_power,
                trend_angle_rad=trend_angle_rad
            )
            synapse.move(det_coordinate[0], det_coordinate[1])
