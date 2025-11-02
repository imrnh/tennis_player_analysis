import argparse
from object_segmented_video_generator import ObjectSegmentatedVideoGenerator
from config import court_marker_config, default_config, det_model_config
from io_utils import write_video

def main():
    parser = argparse.ArgumentParser(description="Generate segmented tennis video.")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save segmented video.")
    args = parser.parse_args()

    # Initialize the video segmentor
    osvg_obj = ObjectSegmentatedVideoGenerator(
        court_marker_model_path=court_marker_config.model_path,
        player_ball_model_path=det_model_config.player_and_ball_model_path,
        net_det_model_path=det_model_config.net_det_model_path,
        default_config=default_config,
        court_marker_config=court_marker_config,
        det_model_config=det_model_config
    )

    # Segment video
    modified_frames, fps = osvg_obj.generate(args.input)

    # Write output
    write_video(modified_frames, fps, args.output)


if __name__ == "__main__":
    main()
