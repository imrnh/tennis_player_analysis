from segmented_video_generator import SegmentatedVideoGenerator
from config import court_marker_config, default_config, det_model_config
from io_utils import write_video


# Initialize the video segmentor.
seg = SegmentatedVideoGenerator(
    court_marker_model_path=court_marker_config.model_path, 
    player_ball_model_path=det_model_config.player_and_ball_model_path,
    net_det_model_path=det_model_config.net_det_model_path,
    default_config=default_config, 
    court_marker_config=court_marker_config,
    det_model_config=det_model_config
)

# Segment the video and write it to folder.
modified_frames, fps = seg.generate("data/tennis_play_record_1_short_v2.mp4")
write_video(modified_frames, fps, "data/segmented_video.avi")