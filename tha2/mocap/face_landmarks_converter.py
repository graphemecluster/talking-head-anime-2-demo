from abc import ABC, abstractmethod
from typing import List


class FaceLandmarksConverter(ABC):
    @abstractmethod
    def convert(self, landmarks: List[List[float]]) -> List[float]:
        pass

    @abstractmethod
    def init_pose_converter_panel(self, parent):
        pass