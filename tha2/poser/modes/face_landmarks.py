import math
from enum import Enum
from typing import List, Optional

import numpy
import wx

from tha2.mocap.face_landmarks_converter import FaceLandmarksConverter
from tha2.poser.modes.mode_20 import get_pose_parameters


def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))


class EyebrowDownMode(Enum):
    TROUBLED = 1
    ANGRY = 2
    LOWERED = 3
    SERIOUS = 4


class WinkMode(Enum):
    NORMAL = 1
    RELAXED = 2


class FaceLandmarksConverter20Args:
    def __init__(self,
                 lower_smile_threshold: float = 0.75,
                 upper_smile_threshold: float = 1.0,
                 head_y_correction: float = 0.075,
                 head_y_zero_value: float = 1.8,
                 head_y_sensitivity: float = 4.0,
                 jaw_open_min_value: float = 0.1,
                 jaw_open_max_value: float = 0.4,
                 eye_blink_min_value: float = 0.2,
                 eye_blink_max_value: float = 0.25,
                 default_sensitivity: float = 5.0,
                 eyebrow_height_min_value: float = 0.0,
                 eyebrow_height_max_value: float = 5.0,
                 eyebrow_down_mode: EyebrowDownMode = EyebrowDownMode.ANGRY,
                 wink_mode: WinkMode = WinkMode.NORMAL,
                 iris_small_left: float = 0.0,
                 iris_small_right: float = 0.0):
        self.lower_smile_threshold = lower_smile_threshold
        self.upper_smile_threshold = upper_smile_threshold
        self.head_y_correction = head_y_correction
        self.head_y_zero_value = head_y_zero_value
        self.head_y_sensitivity = head_y_sensitivity
        self.jaw_open_min_value = jaw_open_min_value
        self.jaw_open_max_value = jaw_open_max_value
        self.eye_blink_min_value = eye_blink_min_value
        self.eye_blink_max_value = eye_blink_max_value
        self.default_sensitivity = default_sensitivity
        self.eyebrow_height_min_value = eyebrow_height_min_value
        self.eyebrow_height_max_value = eyebrow_height_max_value
        self.eyebrow_down_mode = eyebrow_down_mode
        self.wink_mode = wink_mode
        self.iris_small_left = iris_small_left
        self.iris_small_right = iris_small_right

def calculateFaceX(x, y, z):
    return math.sqrt((1 - (x - y - z) ** 2 / (4 * y * z)) * z)

class FaceLandmarksConverter20(FaceLandmarksConverter):
    def __init__(self, args: Optional[FaceLandmarksConverter20Args] = None):
        super().__init__()
        if args is None:
            args = FaceLandmarksConverter20Args()
        self.args = args
        pose_parameters = get_pose_parameters()
        self.pose_size = 42

        self.eyebrow_troubled_left_index = pose_parameters.get_parameter_index("eyebrow_troubled_left")
        self.eyebrow_troubled_right_index = pose_parameters.get_parameter_index("eyebrow_troubled_right")
        self.eyebrow_angry_left_index = pose_parameters.get_parameter_index("eyebrow_angry_left")
        self.eyebrow_angry_right_index = pose_parameters.get_parameter_index("eyebrow_angry_right")
        self.eyebrow_happy_left_index = pose_parameters.get_parameter_index("eyebrow_happy_left")
        self.eyebrow_happy_right_index = pose_parameters.get_parameter_index("eyebrow_happy_right")
        self.eyebrow_raised_left_index = pose_parameters.get_parameter_index("eyebrow_raised_left")
        self.eyebrow_raised_right_index = pose_parameters.get_parameter_index("eyebrow_raised_right")
        self.eyebrow_lowered_left_index = pose_parameters.get_parameter_index("eyebrow_lowered_left")
        self.eyebrow_lowered_right_index = pose_parameters.get_parameter_index("eyebrow_lowered_right")
        self.eyebrow_serious_left_index = pose_parameters.get_parameter_index("eyebrow_serious_left")
        self.eyebrow_serious_right_index = pose_parameters.get_parameter_index("eyebrow_serious_right")

        self.eye_surprised_left_index = pose_parameters.get_parameter_index("eye_surprised_left")
        self.eye_surprised_right_index = pose_parameters.get_parameter_index("eye_surprised_right")
        self.eye_wink_left_index = pose_parameters.get_parameter_index("eye_wink_left")
        self.eye_wink_right_index = pose_parameters.get_parameter_index("eye_wink_right")
        self.eye_happy_wink_left_index = pose_parameters.get_parameter_index("eye_happy_wink_left")
        self.eye_happy_wink_right_index = pose_parameters.get_parameter_index("eye_happy_wink_right")
        self.eye_relaxed_left_index = pose_parameters.get_parameter_index("eye_relaxed_left")
        self.eye_relaxed_right_index = pose_parameters.get_parameter_index("eye_relaxed_right")
        self.eye_raised_lower_eyelid_left_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_left")
        self.eye_raised_lower_eyelid_right_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_right")

        self.iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
        self.iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")

        self.iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
        self.iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")

        self.head_x_index = pose_parameters.get_parameter_index("head_x")
        self.head_y_index = pose_parameters.get_parameter_index("head_y")
        self.neck_z_index = pose_parameters.get_parameter_index("neck_z")

        self.mouth_aaa_index = pose_parameters.get_parameter_index("mouth_aaa")
        self.mouth_iii_index = pose_parameters.get_parameter_index("mouth_iii")
        self.mouth_uuu_index = pose_parameters.get_parameter_index("mouth_uuu")
        self.mouth_eee_index = pose_parameters.get_parameter_index("mouth_eee")
        self.mouth_ooo_index = pose_parameters.get_parameter_index("mouth_ooo")

        self.mouth_lowered_corner_left_index = pose_parameters.get_parameter_index("mouth_lowered_corner_left")
        self.mouth_lowered_corner_right_index = pose_parameters.get_parameter_index("mouth_lowered_corner_right")
        self.mouth_raised_corner_left_index = pose_parameters.get_parameter_index("mouth_raised_corner_left")
        self.mouth_raised_corner_right_index = pose_parameters.get_parameter_index("mouth_raised_corner_right")

    def convert(self, landmarks: List[List[float]]) -> List[float]:
        pose = [0.0 for i in range(self.pose_size)]
        landmarks = numpy.array(landmarks)
        
        eye_diff = numpy.mean(landmarks[36:42], axis=0) - numpy.mean(landmarks[42:48], axis=0)
        smile_value = numpy.linalg.norm(landmarks[48] - landmarks[54]) / numpy.linalg.norm(eye_diff)
        smile_degree = clamp((smile_value - self.args.lower_smile_threshold) / (
                self.args.upper_smile_threshold - self.args.lower_smile_threshold), 0.0, 1.0)
        smile_complement = 1.0 - smile_degree
        pose[self.eyebrow_happy_left_index] = pose[self.eyebrow_happy_right_index] = smile_degree

        y0 = numpy.mean(landmarks[27:31], axis=0)
        y1 = numpy.mean(landmarks[[50, 51, 52, 62], :], axis=0)
        left = numpy.mean(landmarks[14:17], axis=0)
        right = numpy.mean(landmarks[0:3], axis=0)
        side = numpy.sum((y0 - y1) ** 2)
        perp_left = calculateFaceX(numpy.sum((left - y0) ** 2), side, numpy.sum((y1 - left) ** 2))
        perp_right = calculateFaceX(numpy.sum((right - y0) ** 2), side, numpy.sum((y1 - right) ** 2))
        pose[self.head_y_index] = clamp(math.degrees(math.asin((perp_right - perp_left) / (perp_right + perp_left))) / 15.0, -1.0, 1.0)
        
        pose[self.head_x_index] = clamp(((numpy.sum((landmarks[31] - landmarks[35]) ** 2) \
            - numpy.sum((landmarks[30] - landmarks[31]) ** 2) - numpy.sum((landmarks[30] - landmarks[35]) ** 2)) / \
            (-2 * numpy.linalg.norm(landmarks[30] - landmarks[31]) * numpy.linalg.norm(landmarks[30] - landmarks[35])) * \
            (1 + abs(pose[self.head_y_index]) * self.args.head_y_correction) * \
            (1 - smile_degree * self.args.head_y_correction) - self.args.head_y_zero_value) * \
            self.args.head_y_sensitivity, -1.0, 1.0)
        
        nose_diff = landmarks[35] - landmarks[31]
        pose[self.neck_z_index] = clamp(((math.atan(eye_diff[0] / eye_diff[1]) if eye_diff[1] != 0 else 0) + \
            (math.atan(nose_diff[0] / nose_diff[1]) if nose_diff[1] != 0 else 0)) / 30.0, -1.0, 1.0)
        
        eye_correction = 2 * math.cos(pose[self.head_x_index] * math.pi / 12.0)
        eye_blink_left = (numpy.linalg.norm(landmarks[43] - landmarks[47]) + numpy.linalg.norm(landmarks[44] - landmarks[46])) / (numpy.linalg.norm(landmarks[42] - landmarks[45]) * eye_correction)
        eye_blink_right = (numpy.linalg.norm(landmarks[37] - landmarks[41]) + numpy.linalg.norm(landmarks[38] - landmarks[40])) / (numpy.linalg.norm(landmarks[36] - landmarks[39]) * eye_correction)

        pose[self.eye_happy_wink_left_index] = eye_blink_left * smile_degree
        pose[self.eye_happy_wink_right_index] = eye_blink_right * smile_degree
        if self.args.wink_mode == WinkMode.NORMAL:
            pose[self.eye_wink_left_index] = eye_blink_left * smile_complement
            pose[self.eye_wink_right_index] = eye_blink_right * smile_complement
        else:
            pose[self.eye_relaxed_left_index] = eye_blink_left * smile_complement
            pose[self.eye_relaxed_right_index] = eye_blink_right * smile_complement

        unit = numpy.linalg.norm(landmarks[27] - landmarks[28]) / self.args.default_sensitivity
        eye_position_left = numpy.clip((numpy.mean(landmarks[[43, 44, 46, 47], :], axis=0) - numpy.mean(landmarks[[42, 45], :], axis=0)) / unit, -1.0, 1.0)
        eye_position_right = numpy.clip((numpy.mean(landmarks[[37, 38, 40, 41], :], axis=0) - numpy.mean(landmarks[[36, 39], :], axis=0)) / unit, -1.0, 1.0)

        eye_raised_left = max(0.0, eye_position_left[1])
        eye_raised_right = max(0.0, eye_position_right[1])
        pose[self.eye_surprised_left_index] = eye_raised_left * eye_blink_left
        pose[self.eye_surprised_right_index] = eye_raised_right * eye_blink_right
        pose[self.eye_raised_lower_eyelid_left_index] = eye_raised_left
        pose[self.eye_raised_lower_eyelid_right_index] = eye_raised_right
        pose[self.iris_rotation_y_index] = (eye_position_left[0] + eye_position_right[0]) / 2
        pose[self.iris_rotation_x_index] = (eye_position_left[1] + eye_position_right[1]) / 2

        eyebrow_min = (self.args.eyebrow_height_max_value + self.args.eyebrow_height_min_value) / 2
        eyebrow_denom = self.args.eyebrow_height_max_value - eyebrow_min
        eyebrow_left = numpy.clip(((landmarks[24] - numpy.mean(landmarks[[22, 26], :], axis=0)) / unit - eyebrow_min) / eyebrow_denom, -1.0, 1.0)
        eyebrow_right = numpy.clip(((landmarks[19] - numpy.mean(landmarks[[17, 21], :], axis=0)) / unit - eyebrow_min) / eyebrow_denom, -1.0, 1.0)
        pose[self.eyebrow_raised_left_index] = max(0.0, eyebrow_left[1])
        pose[self.eyebrow_raised_right_index] = max(0.0, eyebrow_right[1])
        brow_down_left = -min(0.0, eyebrow_left[1])
        brow_down_right = -min(0.0, eyebrow_right[1])
        if self.args.eyebrow_down_mode == EyebrowDownMode.TROUBLED:
            pose[self.eyebrow_troubled_left_index] = brow_down_left
            pose[self.eyebrow_troubled_right_index] = brow_down_right
        elif self.args.eyebrow_down_mode == EyebrowDownMode.ANGRY:
            pose[self.eyebrow_angry_left_index] = brow_down_left
            pose[self.eyebrow_angry_right_index] = brow_down_right
        elif self.args.eyebrow_down_mode == EyebrowDownMode.LOWERED:
            pose[self.eyebrow_lowered_left_index] = brow_down_left
            pose[self.eyebrow_lowered_right_index] = brow_down_right
        elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
            pose[self.eyebrow_serious_left_index] = brow_down_left
            pose[self.eyebrow_serious_right_index] = brow_down_right

        pose[self.iris_small_left_index] = self.args.iris_small_left
        pose[self.iris_small_right_index] = self.args.iris_small_right
        
        mouth_center = numpy.mean(landmarks[[62, 66], :], axis=0)
        mouth_corner_left = numpy.clip((mouth_center - landmarks[54]) / unit, -1.0, 1.0)
        mouth_corner_right = numpy.clip((mouth_center - landmarks[48]) / unit, -1.0, 1.0)
        pose[self.mouth_raised_corner_left_index] = max(0.0, mouth_corner_left[1])
        pose[self.mouth_raised_corner_right_index] = max(0.0, mouth_corner_right[1])
        pose[self.mouth_lowered_corner_left_index] = -min(0.0, mouth_corner_left[1])
        pose[self.mouth_lowered_corner_right_index] = -min(0.0, mouth_corner_right[1])

        mouth_open = clamp(
                (numpy.linalg.norm(landmarks[62] - landmarks[66]) / numpy.linalg.norm(landmarks[60] - landmarks[64]) - \
                self.args.jaw_open_min_value) / (self.args.jaw_open_max_value - self.args.jaw_open_min_value), 0.0, 1.0)
        pose[self.mouth_aaa_index] = mouth_open * smile_degree
        pose[self.mouth_uuu_index] = mouth_open * smile_complement
        pose[self.mouth_iii_index] = (1.0 - mouth_open) * smile_degree

        return pose

    def init_pose_converter_panel(self, parent):
        self.panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.panel_sizer)
        self.panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.panel, 0, wx.EXPAND)

        if True:
            eyebrow_down_mode_text = wx.StaticText(self.panel, label=" --- Eyebrow Down Mode --- ",
                                                   style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(eyebrow_down_mode_text, 0, wx.EXPAND)

            self.eyebrow_down_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "ANGRY",
                    "TROUBLED",
                    "SERIOUS",
                    "LOWERED",
                ])
            self.eyebrow_down_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.eyebrow_down_mode_choice, 0, wx.EXPAND)
            self.eyebrow_down_mode_choice.Bind(wx.EVT_CHOICE, self.change_eyebrow_down_mode)

            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            wink_mode_text = wx.StaticText(self.panel, label=" --- Wink Mode --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(wink_mode_text, 0, wx.EXPAND)

            self.wink_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "NORMAL",
                    "RELAXED",
                ])
            self.wink_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.wink_mode_choice, 0, wx.EXPAND)
            self.wink_mode_choice.Bind(wx.EVT_CHOICE, self.change_wink_mode)

            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            iris_size_text = wx.StaticText(self.panel, label=" --- Iris Size --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(iris_size_text, 0, wx.EXPAND)

            self.iris_left_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_left_slider, 0, wx.EXPAND)
            self.iris_left_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)

            self.iris_right_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_right_slider, 0, wx.EXPAND)
            self.iris_right_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)
            self.iris_right_slider.Enable(False)

            self.link_left_right_irises = wx.CheckBox(
                self.panel, label="Use same value for both sides")
            self.link_left_right_irises.SetValue(True)
            self.panel_sizer.Add(self.link_left_right_irises, wx.SizerFlags().CenterHorizontal().Border())
            self.link_left_right_irises.Bind(wx.EVT_CHECKBOX, self.link_left_right_irises_clicked)

        self.panel_sizer.Fit(self.panel)

    def change_eyebrow_down_mode(self, event: wx.Event):
        selected_index = self.eyebrow_down_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.eyebrow_down_mode = EyebrowDownMode.ANGRY
        elif selected_index == 1:
            self.args.eyebrow_down_mode = EyebrowDownMode.TROUBLED
        elif selected_index == 2:
            self.args.eyebrow_down_mode = EyebrowDownMode.SERIOUS
        else:
            self.args.eyebrow_down_mode = EyebrowDownMode.LOWERED

    def change_wink_mode(self, event: wx.Event):
        selected_index = self.wink_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.wink_mode = WinkMode.NORMAL
        else:
            self.args.wink_mode = WinkMode.RELAXED

    def change_iris_size(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            left_value = self.iris_left_slider.GetValue()
            right_value = self.iris_right_slider.GetValue()
            if left_value != right_value:
                self.iris_right_slider.SetValue(left_value)
            self.args.iris_small_left = left_value / 1000.0
            self.args.iris_small_right = left_value / 1000.0
        else:
            self.args.iris_small_left = self.iris_left_slider.GetValue() / 1000.0
            self.args.iris_small_right = self.iris_right_slider.GetValue() / 1000.0

    def link_left_right_irises_clicked(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            self.iris_right_slider.Enable(False)
        else:
            self.iris_right_slider.Enable(True)
        self.change_iris_size(event)


def create_face_landmarks_converter(
        args: Optional[FaceLandmarksConverter20Args] = None) -> FaceLandmarksConverter:
    return FaceLandmarksConverter20(args)
