from segmented_video_generator import SegmentatedVideoGenerator

from config import court_marker_config, default_config
from io_utils import write_video


seg = SegmentatedVideoGenerator(court_marker_config.model_path, default_config, court_marker_config)



modified_frames, fps = seg.generate("data/tennis_play_record_1_short.mp4")


print("Writing")
write_video(modified_frames, fps, "data/segmented_video.mp4")