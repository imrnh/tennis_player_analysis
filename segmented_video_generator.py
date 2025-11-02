import torch
import torch.nn.functional as F

import sys
import os
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "court_marker"))


from court_marker import BallTrackerNet
from court_marker import postprocess, refine_kps
from court_marker import get_trans_matrix, refer_kps
from io_utils import read_video, write_video
from config import court_marker_config, default_config
from pose_detector import detect_poses_in_boxes, draw_pose_on_image


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SegmentatedVideoGenerator:
    def __init__(self, court_marker_model_path, 
                 player_ball_model_path, net_det_model_path,
                   default_config, court_marker_config, det_model_config):
        
        # Court Marker Model
        self.court_marker = BallTrackerNet(out_channels=15).to(device)
        self.court_marker.load_state_dict(torch.load(court_marker_model_path, map_location=device))
        self.court_marker.eval()

        # YOLO Models
        self.player_ball_model = YOLO(player_ball_model_path)
        self.net_model = YOLO(net_det_model_path)


        # 4. Class IDs and Confidence threshold.
        self.CONF_THRESHOLD = det_model_config.conf_threshold
        self.PLAYER_CLASS_ID = 0 # Assumed. Need to check dataset before commit.
        self.BALL_CLASS_ID = 1
        self.NET_CLASS_ID = 0 # as this model only detect one class.


        # Configs
        self.default_config = default_config
        self.court_marker_config = court_marker_config
        self.det_model_config = det_model_config

        # Mediapipe setup for pose detection.
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.LANDMARK_DRAWING_SPEC = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.CONNECTION_DRAWING_SPEC = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)



    def generate(self, input_path):
        frames, fps = read_video(input_path)
        self.frames_upd = []
        for image in tqdm(frames):
            image = cv2.resize(image, (default_config.PREPROCESSOR_WIDTH, default_config.PREPROCESSOR_HEIGHT))

            # Create the black image.
            black_image = np.zeros_like(image)

            img = cv2.resize(image, (self.default_config.OUTPUT_WIDTH, self.default_config.OUTPUT_HEIGHT))
            inp = (img.astype(np.float32) / 255.)
            inp = torch.tensor(np.rollaxis(inp, 2, 0))
            inp = inp.unsqueeze(0)

            out = self.court_marker(inp.float().to(device))[0]
            pred = F.sigmoid(out).detach().cpu().numpy()

            # Court marker points detection.
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


            # Predict Player, Ball and Net Detection
            player_ball_results_list = self.player_ball_model(image, verbose=False)
            net_results_list = self.net_model(image, verbose=False)



            # Process and Draw detection            
            player_boxes = []
            ball_boxes = []
            net_boxes = []

            # a. Parse Player/Ball results
            # Get the first (and only) result object from the list as we are operating for each image separately. 
            player_results = player_ball_results_list[0] 
            
            # Iterate over the .boxes attribute
            for box in player_results.boxes:
                conf = box.conf.item()  # Confidence score
                if conf > self.CONF_THRESHOLD:
                    # Get box coordinates
                    xyxy = [int(p) for p in box.xyxy[0]]
                    # Get class id
                    cls = int(box.cls.item())
                    
                    if cls == self.PLAYER_CLASS_ID:
                        player_boxes.append(xyxy)
                    elif cls == self.BALL_CLASS_ID:
                        ball_boxes.append(xyxy)

            # b. Parse Net results
            net_results = net_results_list[0] # Get the first result object
            for box in net_results.boxes:
                conf = box.conf.item()
                cls = int(box.cls.item())
                if conf > self.CONF_THRESHOLD and cls == self.NET_CLASS_ID:
                    net_boxes.append([int(p) for p in box.xyxy[0]])

            # c. Draw Net (Semi-transparent brown) | We do this first so it's "under" the players and court lines
            if net_boxes:
                overlay = black_image.copy()
                for x1, y1, x2, y2 in net_boxes:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                  color=self.default_config.court_net_color, thickness=-1)
                
                alpha = 0.4  # Transparency level
                black_image = cv2.addWeighted(overlay, alpha, black_image, 1 - alpha, 0)


            # d. Draw Player Poses
            if player_boxes:
                # Get pose data using your function
                pose_data = detect_poses_in_boxes(image, player_boxes)
                
                # Draw poses onto the black_image using your function
                black_image = draw_pose_on_image(
                    original_image_shape=image.shape,
                    all_pose_data=pose_data,
                    target_image=black_image
                )

            # e. Draw Ball
            for x1, y1, x2, y2 in ball_boxes:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # Draw a small, filled circle for the ball
                cv2.circle(black_image, (center_x, center_y),
                           radius=5, color=self.default_config.ball_color, thickness=-1)


            # Draw Court Lines | This now draws on top of the net, poses, and ball
            for (i, j) in self.court_marker_config.board_line_pairs:
                if (points[i][0] is not None and points[j][0] is not None):
                    pt1 = (int(points[i][0]), int(points[i][1]))
                    pt2 = (int(points[j][0]), int(points[j][1]))
                    black_image = cv2.line(black_image, pt1, pt2, 
                                           color=self.default_config.court_line_color, thickness=3)

            # Draw Court Points
            for j in range(len(points)):
                if points[j][0] is not None:
                    black_image = cv2.circle(black_image, (int(points[j][0]), int(points[j][1])),
                                       radius=0, color=(0, 0, 255), thickness=10)
                    
            self.frames_upd.append(black_image)
        
        return self.frames_upd, fps


        #     # Connect pairs with line to create the virtual court.
        #     for (i, j) in self.court_marker_config.board_line_pairs:
        #         if (points[i][0] is not None and points[j][0] is not None):
        #             pt1 = (int(points[i][0]), int(points[i][1]))
        #             pt2 = (int(points[j][0]), int(points[j][1]))
        #             black_image = cv2.line(black_image, pt1, pt2, color=(0, 255, 0), thickness=3)


        #     # Draw the points on the screen.
        #     for j in range(len(points)):
        #         if points[j][0] is not None:
        #             black_image = cv2.circle(black_image, (int(points[j][0]), int(points[j][1])),
        #                             radius=0, color=(0, 0, 255), thickness=10)
                    
        #     self.frames_upd.append(black_image)
            
        
        # return self.frames_upd, fps


    