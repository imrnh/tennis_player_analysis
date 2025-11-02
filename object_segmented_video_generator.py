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
from utils import read_video
from pose_detector import draw_pose_on_image, detect_poses_in_boxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ObjectSegmentatedVideoGenerator:
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
        self.PLAYER_CLASS_ID = 1 # Player ID set to 1 in the roboflow dataset.
        self.BALL_CLASS_ID = 0
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

    
    def get_segmentation_predictions(self, image):
        img = cv2.resize(image, (self.default_config.OUTPUT_WIDTH, self.default_config.OUTPUT_HEIGHT))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        out = self.court_marker(inp.float().to(device))[0]
        pred = F.sigmoid(out).detach().cpu().numpy()

        return pred


    def get_court_points(self, image, pred):
        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            if self.court_marker_config.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
            points.append((x_pred, y_pred))

        # Homographic Transformation.
        if self.court_marker_config.use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points]

        return points
    
    def get_player_and_ball_list(self, image, verbose=False):
        player_boxes = []
        ball_boxes = []

        player_ball_results_list = self.player_ball_model(image, verbose=False)
    
        # a. Parse Player/Ball results
        # Get the first (and only) result object from the list as we are operating for each image separately. 
        player_results = player_ball_results_list[0] 
        for box in player_results.boxes: 
            conf = box.conf.item() 
            if conf > self.CONF_THRESHOLD:

                xyxy = [int(p) for p in box.xyxy[0]] # Get box coordinates
                cls = int(box.cls.item()) # Get class id
                
                if cls == self.PLAYER_CLASS_ID:
                    player_boxes.append(xyxy)
                elif cls == self.BALL_CLASS_ID:
                    ball_boxes.append(xyxy)

        
        return player_boxes, ball_boxes
    
    def get_net_list(self, image, verbose=False):
        net_boxes = []
        net_results_list = self.net_model(image, verbose=False)

        # b. Parse Net results
        net_results = net_results_list[0] # Get the first result object
        for box in net_results.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf > self.CONF_THRESHOLD and cls == self.NET_CLASS_ID:
                net_boxes.append([int(p) for p in box.xyxy[0]])

        return net_boxes
    

    def draw_net_(self, black_image, net_boxes):
        # Draw Net. We do this first so it's "under" the players and court lines
        if net_boxes:
            overlay = black_image.copy()
            for x1, y1, x2, y2 in net_boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                color=self.default_config.court_net_color, thickness=-1)
            
            alpha = 0.4  # Transparency level
            black_image = cv2.addWeighted(overlay, alpha, black_image, 1 - alpha, 0)

        return black_image
    

    def draw_player_pose(self, image, black_image, player_boxes):
        # Draw Player Poses onto the black_image
        if player_boxes:
            pose_data = detect_poses_in_boxes(self.mp_pose, image, player_boxes)
            black_image = draw_pose_on_image(
                mp_drawing = self.mp_drawing,
                mp_pose= self.mp_pose,
                LANDMARK_DRAWING_SPEC = self.LANDMARK_DRAWING_SPEC,
                CONNECTION_DRAWING_SPEC= self.CONNECTION_DRAWING_SPEC,
                original_image_shape=image.shape,
                all_pose_data=pose_data,
                target_image=black_image
            )
        return black_image
            
    def draw_ball(self, black_image, ball_boxes):
        for x1, y1, x2, y2 in ball_boxes:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            cv2.circle(black_image, (center_x, center_y), radius=5, color=self.default_config.ball_color, thickness=-1) # Draw a small, filled circle for the ball

        return black_image
    

    def draw_court_marker(self, black_image, marker_points, draw_point=True):
        # Draw Court Lines | This now draws on top of the net, poses, and ball
        for (i, j) in self.court_marker_config.board_line_pairs:
            if (marker_points[i][0] is not None and marker_points[j][0] is not None):
                pt1 = (int(marker_points[i][0]), int(marker_points[i][1]))
                pt2 = (int(marker_points[j][0]), int(marker_points[j][1]))
                black_image = cv2.line(black_image, pt1, pt2, 
                                        color=self.default_config.court_line_color, thickness=3)

        if draw_point: # Draw Court marker_points
            for j in range(len(marker_points)):
                if marker_points[j][0] is not None:
                    black_image = cv2.circle(black_image, (int(marker_points[j][0]), int(marker_points[j][1])),
                                        radius=0, color=(0, 0, 255), thickness=10)
                    
        return black_image



    def generate(self, input_path):
        frames, fps = read_video(input_path)
        self.frames_upd = []
        for image in tqdm(frames):
            image = cv2.resize(image, (self.default_config.PREPROCESSOR_WIDTH, self.default_config.PREPROCESSOR_HEIGHT))
            black_image = np.zeros_like(image) # Create the black image.

            pred = self.get_segmentation_predictions(image)  # Get segmentation mask to get court points.

            marker_points = self.get_court_points(image, pred) # Court marker points detection.
            player_boxes, ball_boxes = self.get_player_and_ball_list(image) # Player and Ball boxes
            net_boxes =  self.get_net_list(image) # Net boxes.

            # Draw Court marker, Net, player, ball
            black_image = self.draw_court_marker(black_image, marker_points, draw_point=True)
            black_image = self.draw_net_(black_image, net_boxes)
            black_image = self.draw_player_pose(image, black_image, player_boxes)
            black_image = self.draw_ball(black_image, ball_boxes)
            

            self.frames_upd.append(black_image)
        
        return self.frames_upd, fps