import torch
import torch.nn.functional as F

import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "court_marker"))


from court_marker import BallTrackerNet
from court_marker import postprocess, refine_kps
from court_marker import get_trans_matrix, refer_kps
from io_utils import read_video, write_video
from config import court_marker_config, default_config


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SegmentatedVideoGenerator:
    def __init__(self, court_marker_model_path, default_config, court_marker_config):
        self.court_marker = BallTrackerNet(out_channels=15).to(device)
        self.court_marker.load_state_dict(torch.load(court_marker_model_path, map_location=device))
        self.court_marker.eval()

        self.default_config = default_config
        self.court_marker_config = court_marker_config

    def generate(self, input_path):
        frames, fps = read_video(input_path)
        print("Frame count:", len(frames), "\t fps:", fps)
        self.frames_upd = []
        for image in tqdm(frames):
            image = cv2.resize(image, (default_config.PREPROCESSOR_WIDTH, default_config.PREPROCESSOR_HEIGHT))
            img = cv2.resize(image, (self.default_config.OUTPUT_WIDTH, self.default_config.OUTPUT_HEIGHT))
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0)

            out = self.court_marker(inp.float().to(device))[0]
            pred = F.sigmoid(out).detach().cpu().numpy()

            points = []
            for kps_num in range(14):
                heatmap = (pred[kps_num] * 255).astype(np.uint8)
                x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
                if self.court_marker_config.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                    x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
                points.append((x_pred, y_pred))


            if self.court_marker_config.use_homography:
                matrix_trans = get_trans_matrix(points)
                if matrix_trans is not None:
                    points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                    points = [np.squeeze(x) for x in points]


            # Connect pairs with line to create the virtual court.
            for (i, j) in self.court_marker_config.board_line_pairs:
                if (points[i][0] is not None and points[j][0] is not None):
                    pt1 = (int(points[i][0]), int(points[i][1]))
                    pt2 = (int(points[j][0]), int(points[j][1]))
                    image = cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=3)


            # Draw the points on the screen.
            for j in range(len(points)):
                if points[j][0] is not None:
                    image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                                    radius=0, color=(0, 0, 255), thickness=10)
                    
            self.frames_upd.append(image)
            
        
        return self.frames_upd, fps


    