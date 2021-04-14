import os
import sys
import time

sys.path.append(os.getcwd())

import numpy
import torch
import wx
import PIL.Image
import pyvirtualcam
import cv2
import dlib
import face_recognition_models

from tha2.poser.poser import Poser
from tha2.mocap.face_landmarks_converter import FaceLandmarksConverter
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy


face_detector = dlib.get_frontal_face_detector()
pose_predictor = dlib.shape_predictor(face_recognition_models.pose_predictor_model_location())

class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, pose_converter: FaceLandmarksConverter, device: torch.device):
        super().__init__(None, wx.ID_ANY, "Webcam Face Tracking")
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device

        self.wx_source_image = None
        self.torch_source_image = None
        self.source_image_string = "Nothing yet!"

        self.main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)
        self.input_panel = wx.Panel(self, size=(256, 312), style=wx.SIMPLE_BORDER)
        self.input_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.input_panel.SetSizer(self.input_panel_sizer)
        self.input_panel.SetAutoLayout(1)
        self.main_sizer.Add(self.input_panel, 0, wx.FIXED_MINSIZE)

        self.source_image_panel = wx.Panel(self.input_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
        self.input_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

        self.load_image_button = wx.Button(self.input_panel, wx.ID_ANY, "Load Image")
        self.input_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
        self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

        self.input_panel_sizer.Fit(self.input_panel)

        self.pose_converter.init_pose_converter_panel(self)

        self.right_panel = wx.Panel(self, style=wx.SIMPLE_BORDER)
        right_panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(right_panel_sizer)
        self.right_panel.SetAutoLayout(1)
        self.main_sizer.Add(self.right_panel, 0, wx.EXPAND)

        self.result_image_panel = wx.Panel(self.right_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
        self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
        self.output_index_choice = wx.Choice(
            self.right_panel,
            choices=[str(i) for i in range(self.poser.get_output_length())])
        self.output_index_choice.SetSelection(0)
        right_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)
        right_panel_sizer.Add(self.output_index_choice, 0, wx.EXPAND)

        self.fps_text = wx.StaticText(self.right_panel, label="")
        right_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())
        right_panel_sizer.Fit(self.right_panel)

        self.main_sizer.Fit(self)

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_panel, self.timer)
        self.timer.Start(30)

        self.last_pose = None
        self.last_output_index = self.output_index_choice.GetSelection()
        self.last_output_numpy_image = None
        self.camera = pyvirtualcam.Camera(256, 256, 60)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def on_close(self, event: wx.Event):
        self.timer.Stop()
        self.capture.release()
        cv2.destroyAllWindows()
        event.Skip()

    def load_image(self, event: wx.Event):
        dir_name = "data/illust"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            try:
                pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file_name))
                w, h = pil_image.size
                self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image).to(self.device)

                self.Refresh()
            except:
                message_dialog = wx.MessageDialog(self, "Could not load image " + image_file_name, "Poser", wx.OK)
                message_dialog.ShowModal()
                message_dialog.Destroy()
        file_dialog.Destroy()

    def paint_source_image_panel(self, event: wx.Event):
        if self.wx_source_image is None:
            self.draw_source_image_string(self.source_image_panel, use_paint_dc=True)
        else:
            dc = wx.PaintDC(self.source_image_panel)
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)

    def paint_result_image_panel(self, event: wx.Event):
        self.last_pose = None

    def draw_source_image_string(self, widget, use_paint_dc: bool = True):
        if use_paint_dc:
            dc = wx.PaintDC(widget)
        else:
            dc = wx.ClientDC(widget)
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent(self.source_image_string)
        dc.DrawText(self.source_image_string, 128 - w // 2, 128 - h // 2)

    def update_result_image_panel(self, event: wx.Event):
        tic = time.perf_counter()

        if self.torch_source_image is None:
            current_pose = None
        else:
            ret, frame = self.capture.read()
            locations = face_detector(frame, 1)
            if len(locations) == 0:
                return
            landmarks = [[p.x, p.y] for p in pose_predictor(frame, locations[0]).parts()]
            current_pose = self.pose_converter.convert(landmarks)

        if self.last_pose is not None \
                and self.last_pose == current_pose \
                and self.last_output_index == self.output_index_choice.GetSelection():
            return
        self.last_pose = current_pose
        self.last_output_index = self.output_index_choice.GetSelection()

        if self.torch_source_image is None:
            self.draw_source_image_string(self.result_image_panel, use_paint_dc=False)
            return

        pose = torch.tensor(current_pose, dtype=torch.float, device=self.device)
        output_index = self.output_index_choice.GetSelection()
        output_image = self.poser.pose(self.torch_source_image, pose, output_index)[0].detach().cpu()
        numpy_image = convert_output_image_from_torch_to_numpy(output_image)
        self.last_output_numpy_image = numpy_image
        wx_image = wx.ImageFromBuffer(
            numpy_image.shape[0],
            numpy_image.shape[1],
            numpy_image[:, :, 0:3].tobytes(),
            numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.ClientDC(self.result_image_panel)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap, (256 - numpy_image.shape[0]) // 2, (256 - numpy_image.shape[1]) // 2, True)
        self.camera.send(numpy_image)

        toc = time.perf_counter()
        elapsed_time = toc - tic
        fps = 1.0 / elapsed_time
        self.fps_text.SetLabelText("FPS = %0.2f" % fps)

if __name__ == "__main__":
    import tha2.poser.modes.face_landmarks
    import tha2.poser.modes.mode_20

    cuda = torch.device('cuda')
    poser = tha2.poser.modes.mode_20.create_poser(cuda)
    pose_converter = tha2.poser.modes.face_landmarks.create_face_landmarks_converter()

    app = wx.App()
    main_frame = MainFrame(poser, pose_converter, cuda)
    main_frame.Show(True)
    app.MainLoop()
