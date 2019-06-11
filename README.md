# Hand_Keypoint_Detection
reference paper : http://openaccess.thecvf.com/content_cvpr_2017/papers/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.pdf

1. The Network Architecture
![Network architecture](/Image_for_github/Network.png)

2. Dataset
To train this network, I use MPII Hand keypoint dataset.
Datset Information is here.
In annotation file, there is 21 keypoints information. And we could get 20 limb data by using 21 keypoints.
![Dataset_Information](/Image_for_github/Dataset_Information.png)

3. Training

![Training Loss](/Image_for_github/Train_Loss.png)

The upper graph is loss of confidence map, and the lower graph is loss of PAFs.


4. Output
I could get these output.
First is image with representing keypoint.

![Confidencemap](/Image_for_github/Confidencemap.PNG)

And the second is image with representing PAFs with arrows.

![PAFs](/Image_for_github/PAFs.PNG)

The Confidencemap's size is [44x44x22]. And PAFs's size is [44x44x44].
So, I could get 22 confidencemaps and 44PAFs.
Like this.

![Confidencemap_PAFs_Hands](./Image_for_github/Hand.png)

And, finally we could get these output
![Confidencemap_output](/Image_for_github/Hand_Output2.PNG)
![Confidencemap_output](/Image_for_github/Hand_Output.PNG)


5. etc
Information about this repository.

1) Demo_Hand.ipynb : If you trained this network, you could get PAFs and Confidencemaps by running session.
Then, you use this ipynb file to estimate actual pose by using bipartite argorithm.

2) For_Demo_Hand.ipynb : This file is for Demo_Hand.ipynb. 

3) Hand_Data_Processing.ipynb : You download MPII Hand keypoint dataset, and you could get raw annotation data not good for using.
So, I made this file for using annotation data easily.

4) Training_Hand.ipynb : This file is for training.
