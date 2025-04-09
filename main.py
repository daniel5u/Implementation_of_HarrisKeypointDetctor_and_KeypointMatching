"""
Directory Structure:
--------------------
your_project/
├── images/
│   ├── left.jpg         # Input left image.
│   └── right.jpg        # Input right image.
├── features.py          # Contains keypoint detectors, descriptors, and matchers.
├── transformation.py    # Additional module
└── main.py              # Main script: loads images, processes features, and saves results.

Output images (e.g., left_harris.jpg, right_harris.jpg, matches.jpg) need be saved in the images/ folder.

Fixed Parameters:
-----------------
- HARRIS_THRESHOLD = 0.08     : Only keypoints with a Harris response greater than 0.08 will be kept.
- HARRIS_KERNEL    = 7        : Use a 7-pixel kernel for non-maximum suppression.
- MATCH_RATIO_THRESH = 0.9   : Ratio threshold for feature matching.
- Drawing parameters:
    - GREEN = (0, 255, 0)      : Color for drawing.
    - RADIUS = 5               : Radius for drawing circles at keypoint locations.
"""

import cv2
import numpy as np
import os
from scipy import spatial
from features import HarrisKeypointDetector, MOPSFeatureDescriptor, SSDFeatureMatcher, ORBKeypointDetector, ORBFeatureMatcher, ORBFeatureDescriptor

# Your TunedHarrisDetector should extend HarrisKeypointDetector and apply non-max suppression using HARRIS_KERNEL.
class TunedHarrisDetector(HarrisKeypointDetector):
    def detectKeypoints(self, img):
        HARRIS_THRESHOLD = 0.08
        # 1. Call the base detector to get keypoints.
        keypoints = super().detectKeypoints(img)

        # 2. Convert keypoints to an array with (x, y, response).
        keypoints_len = len(keypoints)
        keypoints2array=np.zeros((keypoints_len,3))
        
        for i in range(keypoints_len):
            keypoints2array[i,0],keypoints2array[i,1] = keypoints[i].pt
            keypoints2array[i,2] = keypoints[i].response
        
        # 3. Sort by response and apply non-maximum suppression using HARRIS_KERNEL.
        sorted_indices = np.argsort(keypoints2array[:, 2])  # 获取第三列的索引排序
        keypointsArraySorted = keypoints2array[sorted_indices]  # 依据排序后的索引重排数组
        keypointsSorted = [keypoints[i] for i in sorted_indices]
        '''
        我寻思non-maximum不是在detectKeypoints里已经实现了吗。。。？
        '''
        # 4. Return only keypoints with response > HARRIS_THRESHOLD.
        for j in range(keypoints_len):
            # print(keypointsSorted[j].response)
            if keypointsSorted[j].response > HARRIS_THRESHOLD:
                break
        return keypointsSorted[j:]
        pass  # Fill in your implementation.

# Your RatioTestMatcher should perform ratio test feature matching.

class RatioTestMatcher:
    MATCH_RATIO_THRESH = 0.9
    def matchFeatures(self, desc1, desc2, ratio_thresh=MATCH_RATIO_THRESH):
        # 1. Compute the Euclidean distance matrix between desc1 and desc2.
        # 2. For each descriptor, find the closest two matches and apply a ratio test with ratio_thresh.
        '''
        到底要干嘛？ratiofeaturematch能不能用？？？
        '''
        distances = spatial.distance.cdist(desc1,desc2,'euclidean')
        indices = np.argmin(distances,axis=1)   #对于每一个desc1，desc2中哪一个和他的距离最小

        amount = desc1.shape[0]
        distances_sorted_within_one_row = np.sort(distances)
        sort_indices = np.argsort(distances_sorted_within_one_row[:,0])
        distances_sorted_both = distances_sorted_within_one_row[sort_indices]
        two_nearest = distances_sorted_both[:,:2]
        matches = []

        ratio_distance = np.where(two_nearest[:,1] != 0,two_nearest[:,0] / two_nearest[:,1] , 0)
        for i in range(amount):
            if ratio_distance[i] < ratio_thresh:
                matches.append(cv2.DMatch(sort_indices[i],indices[sort_indices[i]],distances_sorted_both[i,0]))
        #接下来再对于matches排序，选出最匹配的几组
        return matches
        pass  # Fill in your implementation.

# Drawing functions: draw keypoints and match lines with circles.
def draw_harris_keypoints(image, keypoints, output_path):
    keypoints_len = len(keypoints)
    image_dup = np.copy(image)
    for i in range(keypoints_len):
        cv2.drawMarker(image_dup,(int(keypoints[i].pt[0]),int(keypoints[i].pt[1])),color=(0,255,0))
    
    cv2.imwrite(output_path,image_dup)
    # For each keypoint, draw a cross marker using cv2.drawMarker.
    pass  # Fill in your implementation.


def draw_matches(img1, kp1, img2, kp2, matches, output_path, max_matches=50):
    GREEN = (0, 255, 0)
    RADIUS = 5  
    # Concatenate images, then for each match draw a line and circles (radius fixed to RADIUS).
    combined_array = np.hstack((img1, img2))
    width = img1.shape[1]
    matches_len = len(matches)
    print("Total match amount:",matches_len)
    for i in range(matches_len):
        if i < max_matches:
            pt1 = kp1[matches[i].queryIdx]
            pt2 = kp2[matches[i].trainIdx]
            x1,y1 = pt1.pt
            x2,y2 = pt2.pt
            cv2.line(combined_array, (int(x1),int(y1)), (int(x2+width),int(y2)),color=(0,0,255))
            cv2.circle(combined_array,(int(x1),int(y1)),RADIUS,GREEN)
            cv2.circle(combined_array,(int(x2+width),int(y2)),RADIUS,GREEN)
        else:
            break
    cv2.imwrite(output_path,combined_array)
    pass  # Fill in your implementation.


if __name__ == "__main__":
    # 1. Initialize your modules (TunedHarrisDetector, MOPSFeatureDescriptor, RatioTestMatcher).
    HD = HarrisKeypointDetector()
    THD = TunedHarrisDetector() #from image to keypoints
    ORB_key = ORBKeypointDetector()
    MOP = MOPSFeatureDescriptor() #from keypoints to desc
    ORB_desc = ORBFeatureDescriptor()
    RTM = RatioTestMatcher() #from desc to mathes
    ORB_FM = ORBFeatureMatcher()
    # 2. Read left.jpg and right.jpg from the images folder.
    imageL = cv2.imread("images/left.jpg")
    imageR = cv2.imread("images/right.jpg")

    # 3. Detect keypoints, compute descriptors, and match features.
    keyL = HD.detectKeypoints(imageL)
    keyR = HD.detectKeypoints(imageR)
    # keyL_ORB = ORB_key.detectKeypoints(imageL)
    # keyR_ORB = ORB_key.detectKeypoints(imageR)

    descL = MOP.describeFeatures(imageL,keyL)
    descR = MOP.describeFeatures(imageR,keyR)
    # descL_ORB = MOP.describeFeatures(imageL,keyL_ORB)
    # descR_ORB = MOP.describeFeatures(imageR,keyR_ORB)

    match = RTM.matchFeatures(descL,descR)
    # match_orb = RTM.matchFeatures(descL_ORB,descR_ORB)


    # 4. Draw and save the keypoints and match results.
    draw_harris_keypoints(imageL,keyL,"images/left_harris.jpg")
    draw_harris_keypoints(imageR,keyR,"images/right_harris.jpg")
    draw_matches(imageL,keyL,imageR,keyR,match,"images/matches.jpg")