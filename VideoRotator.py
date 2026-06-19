#   this code ROTATES (and ZOOMS) the video relative to the mouse position

import py3r.behaviour as p3b
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
import os


class VideoRotator:

    def __init__(self, video_path : str, output_path : str, out_width : int, out_height : int):
        """ 
        Docstring for __init__
        
        :param self: Description
        :param video_path: Description
        :type video_path: str
        :param output_path: Description
        :type output_path: str
        :param zoom: how much you want to zoom, None = 1; 5 means a X5 zoom.
        :type zoom: float
        """

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise KeyError(f"{video_path} is not a valid video path")
        
        self.video_path = video_path
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (out_width, out_height), isColor=False)
        if not self.out.isOpened():
            raise KeyError(f"{output_path} is not a valid output path")

        self.crop_x0 = (self.original_width - out_width) // 2
        self.crop_y0 = (self.original_height - out_height) // 2
        self.crop_x1 = self.crop_x0 + out_width
        self.crop_y1 = self.crop_y0 + out_height


    def follow(self, tracking_path : str, centre : str, endpoint : str):

        tracking = p3b.Tracking.from_yolo3r(filepath = tracking_path, fps=self.fps, handle = "video")
        tracking.filter_likelihood(threshold=0.7)
        tracking.interpolate(limit=5)
        tracking.smooth_all(method="savgol")
        tracking_dataframe = tracking.data

        with tqdm(desc = f"rotating {self.video_path}", total = self.total_frames, ascii = True) as pbar:
            for n in range(0, self.total_frames):
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    gray_frame = np.zeros(
                        (self.original_height, self.original_width),
                        dtype=np.uint8
                    )
                    print(f"frame {n} not read was imputed")

                px = tracking_dataframe.loc[n, endpoint+".x"]
                py = tracking_dataframe.loc[n, endpoint+".y"]
                cx = tracking_dataframe.loc[n, centre+".x"]
                cy = tracking_dataframe.loc[n, centre+".y"]

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                tx = -(cx - self.original_width/2)
                ty = -(cy - self.original_height/2)
                traslation_matrix = np.float32([
                                                 [1, 0, tx],
                                                 [0, 1, ty]
                                                 ])
                traslated_frame = cv2.warpAffine(gray_frame, traslation_matrix,(self.original_width, self.original_height))

                angle = math.degrees(math.atan2(py - cy, px - cx)) + 90
                rotation_matrix = cv2.getRotationMatrix2D((self.original_width / 2, self.original_height / 2), angle, 1 )
                rotated_frame = cv2.warpAffine(traslated_frame, rotation_matrix, (self.original_width, self.original_height))

                cropped_frame = rotated_frame[self.crop_y0:self.crop_y1, self.crop_x0:self.crop_x1]

                p_low, p_high = np.percentile(cropped_frame, (1, 99))
                clipped = np.clip(cropped_frame, p_low, p_high)
                normalized_frame = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                self.out.write(normalized_frame)
                pbar.update(1)
        self.close()

    def close(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
