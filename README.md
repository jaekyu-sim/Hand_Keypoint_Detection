# Hand_Keypoint_Detection
reference paper : http://openaccess.thecvf.com/content_cvpr_2017/papers/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.pdf

1. The Network Architecture


2. Dataset

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
![Confidencemap_output](/Image_for_github/Hand_Output1.PNG)
![Confidencemap_output](/Image_for_github/Hand_Output2.PNG)


5. etc
