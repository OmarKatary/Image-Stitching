# Image-Stitching
### Overview
After choosing corresponding points manually, they were then used to compute the homography matix. This matrix is used to inversely warp each pixel of the second image onto the first image. This was done for every colour channel and was interpolated into position using linear interpolation.
### Results
 <img src="https://raw.githubusercontent.com/OmarKatary/Image-Stitching/master/images/result.PNG" alt="drawing" width="380"/>
