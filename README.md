# K-Means-Clustering-of-Video-Framers-for-Pre-Processing-
This program detects a target-colored object in video frames, crops a centered region, and converts it to HSV histogram features. Frames are clustered with k-means++ using silhouette scoring, and the most representative frame from each cluster is saved as a keyframe summary.

1.	Place your video file (e.g., my_video.mp4) in the same folder as the program or update the video_path variable with its location.
2.	Run the program. It will process the video, detect the target-colored object in sampled frames, and extract centered crops.
3.	The program will automatically cluster frames using k-means++ and select the most representative frames from each cluster.
4.	Keyframes will be saved in a newly created folder (e.g., keyframes0, keyframes1, etc.) in the same directory.
5.	Open the folder to view the extracted keyframes that summarize your video.
