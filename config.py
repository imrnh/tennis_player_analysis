from types import SimpleNamespace



default_config = SimpleNamespace(
    OUTPUT_WIDTH = 640,
    OUTPUT_HEIGHT = 360,

    PREPROCESSOR_WIDTH = 1280,
    PREPROCESSOR_HEIGHT = 720,
)


# Court Marker Config
court_marker_config = SimpleNamespace(
    model_path = "pretrained_models/model_tennis_court_det.pt",
    use_refine_kps = True,
    use_homography = False,

    # Pair to build the board.
    board_line_pairs = [(0, 4), (4, 6), (6, 1), (0, 2), (1, 3), (4, 8),
              (8, 10), (10, 5), (9, 11), (6, 9), (11, 7), (12, 13), (2, 5),
              (5, 7), (7, 3), (10, 13), (13, 11), (8, 12), (12, 9)],

    # Color of the net and the ball.    
    court_net_color = (42, 42, 165),
    ball_color = (0, 255, 255),
    court_line_color=(255, 255, 255)
)

# Model path
det_model_config = SimpleNamespace(
    player_and_ball_model_path = "pretrained_models/player_and_ball_detection_best.pt",
    net_det_model_path = "pretrained_models/net_detection_best.pt",

    conf_threshold = 0.5,

)