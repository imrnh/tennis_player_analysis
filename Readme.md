# Tennis Serve Extraction and Analysis 

### Extracted the following informations:

From a tennis gameplay video, 

1) Identify a serve and extract the type for each of the serves (Flat / Kick / Slice) 
2) Determine the success of the Serve (in/out/let)
3) Calculate *toss height*, *toss position*, *hit height*, *serve speed* and *player speed*.


# 1) Run
You can either run the whole pipeline to get the stats of a player. Or you can separetely run each of the step in workflow. We will start with seperate workflow first. If you need to run the whole pipeline at once, please check the bottom of the file.

## 1.1) Get Segmented Video 
The segmented video is used to train the model and then inference from it. This helps the model to only focus on relevant information and hence drastically reduces the data requirements. The following comamnd generates the segmented output.

```
!python make_segmentation.py \
  --input bin/data/inference/tennis_play_record_1_short_v2.mp4 \
  --output bin/data/output_segmented_video.avi
```