# K-Means-Clustering-of-Video-Framers-for-Pre-Processing-
This program detects a target-colored object in video frames, crops a centered region, and converts it to HSV histogram features. Frames are clustered with k-means++ using silhouette scoring, and the most representative frame from each cluster is saved as a keyframe summary.

1.	Place your video file (e.g., my_video.mp4) in the same folder as the program or update the video_path variable with its location.
2.	Run the program. It will process the video, detect the target-colored object in sampled frames, and extract centered crops.
3.	The program will automatically cluster frames using k-means++ and select the most representative frames from each cluster.
4.	Keyframes will be saved in a newly created folder (e.g., keyframes0, keyframes1, etc.) in the same directory.
5.	Open the folder to view the extracted keyframes that summarize your video.

EXAMPLE OF KEYFRAMES

![cluster_7](https://github.com/user-attachments/assets/5cf2a52b-78c4-4c48-960d-45f8bae705b1)
![cluster_6](https://github.com/user-attachments/assets/77e6710f-e6cb-4466-9b9d-822d8431e0c0)
![cluster_5](https://github.com/user-attachments/assets/8b9875e6-8a2d-405b-9003-e9658b6e4761)
![cluster_4](https://github.com/user-attachments/assets/a7f10600-6bf8-4fca-9c29-5354f7028f85)
![cluster_3](https://github.com/user-attachments/assets/3a4c25c6-6266-42b1-9eb0-50b060808037)
![cluster_2](https://github.com/user-attachments/assets/b4df1cee-f8d7-4b7c-95f8-5422bc514d0d)
![cluster_1](https://github.com/user-attachments/assets/24b0257c-c89d-4a53-a5d9-914e671f215b)
