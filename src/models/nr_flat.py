# -*- coding: utf-8 -*-

# Near Real - Flat

import numbers
import types
import uuid
from abc import abstractmethod, ABCMeta
from __future__ import annotations

# 一个节点有可能在在一个tick内接受

class PathNode(object, metaclass=ABCMeta):
    def __default_activate_function(value: numbers.Number) -> numbers.Number:
        if  value <= 0:
            return 0
        else:
            return 1

    nid = None
    coordinate = []
    neighbors = []
    activate_function = None

    def get_nid(self) -> uuid.UUID:
        return self.nid

    def get_coordinate(self) -> list[numbers.Number]:
        return self.coordinate

    def get_neighbors(self) -> list[PathNode]:
        return self.neighbors

    def get_activate_function(self) -> types.FunctionType[numbers.Number]:
        return self.activate_function

    @abstractmethod
    def try_active(self, value: numbers.Number) -> None:
        result = self.activate_function(value)
        if(result > 0):
            self.active(result)

    @abstractmethod
    def active(self, value: numbers.Number) -> None:
        for neighbor in self.neighbors:
            neighbor.try_active(value)


class Neuron(PathNode):
    def __init__(self):
        super(Neuron, self).__init__()
        pass

