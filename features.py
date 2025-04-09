import math
import cv2
import numpy as np
from scipy import ndimage, spatial
import transformations

def inbounds(shape, indices):
    """
    Check if the given indices are within the bounds of an array shape.
    
    Input:
      - shape: tuple, e.g. (rows, cols)
      - indices: tuple, e.g. (row_index, col_index)
    Output:
      - Boolean: True if indices are valid, otherwise False.
    """
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        """
        Input:
          - image: uint8 BGR image with pixel values in [0, 255].
        Output:
          - A list of detected keypoints. Each keypoint should be a cv2.KeyPoint object with:
              • pt: (x, y) coordinates,
              • angle: gradient orientation (in degrees),
              • response: detector response (e.g. Harris score),
              • size: set to 10.
        """

    
class DummyKeypointDetector(KeypointDetector):
    """
    A silly detector that generates dummy keypoints based on an arbitrary condition.
    """
    def detectKeypoints(self, image):
        """
        Input:
          - image: uint8 BGR image.
        Output:
          - List of cv2.KeyPoint objects.
        Hint:
          - Iterate over every pixel.
          - Use a simple condition (e.g., based on the sum of the channels modulo a constant)
            to decide whether to create a keypoint.
        """
        # TODO: Implement dummy keypoint detection according to the hint.
        image_float32 = image.astype(np.float32) / 255
        image_gray = cv2.cvtColor(image_float32,cv2.COLOR_BGR2GRAY)
        harris,grad = self.computeHarrisValues(image_gray)

        height,width = image.shape

        keypoints = []

        for i in range(height):
            for j in range(width):
              if (image[i][j][0]+image[i][j][1]+image[i][j][2]) % 2 == 1:
                  temp_keypoint = cv2.KeyPoint(x=j, y=i, angle=grad[i][j], response=harris[i][j], size=10)
                  keypoints.append(temp_keypoint)
        return keypoints

class HarrisKeypointDetector(KeypointDetector):
    def computeHarrisValues(self, srcImage):
        """
        Input:
          - srcImage: Grayscale image (float32) with values in [0, 1], shape (is, cols).
        Output:
          - harrisImage: numpy array of the same shape containing the Harris corner strength.
          - orientationImage: numpy array (same shape) containing gradient orientation (in degrees).
        Parameter hints:
          - Use ndimage.sobel to compute image gradients Ix and Iy.
          - Compute Ixx, Ixy, Iyy, and smooth them with a Gaussian filter.
          - Compute the determinant (det = Ixx * Iyy - Ixy^2) and trace (Ixx + Iyy).
          - Use a formula (e.g., det - k*(trace)^2 or a variant) to get the response.
          - Calculate gradient orientation using np.arctan2(Iy, Ix) and convert to degrees.
        """
        # TODO: Compute Harris corner responses and gradient orientations.

        sobel_x = ndimage.sobel(srcImage,axis=0,mode='reflect')
        sobel_y = ndimage.sobel(srcImage,axis=1,mode='reflect')

        sobel_x2 = np.square(sobel_x)
        sobel_y2 = np.square(sobel_y)
        sobel_xy = np.multiply(sobel_x,sobel_y)

        sobel_x2_gaussian = cv2.GaussianBlur(sobel_x2,ksize=(5,5),sigmaX=0.5)
        sobel_y2_gaussian = cv2.GaussianBlur(sobel_y2,ksize=(5,5),sigmaX=0.5)
        sobel_xy_gaussian = cv2.GaussianBlur(sobel_xy,ksize=(5,5),sigmaX=0.5)

        height,width = srcImage.shape

        Harris_response = np.zeros((height,width), np.float32)
        grad_orien = np.zeros((height,width), np.float32)
        det = sobel_x2_gaussian * sobel_y2_gaussian - sobel_xy_gaussian
        trace = sobel_x2_gaussian + sobel_y2_gaussian
        R = det - 0.1 * trace**2
        for i in range(height):
            for j in range(width):
                Harris_response[i][j] = R[i][j]
                grad_orien[i][j] = np.arctan2(sobel_y[i][j], sobel_x[i][j]) * 180 / np.pi

        return Harris_response,grad_orien

    def computeLocalMaxima(self, harrisImage):
        """
        Input:
          - harrisImage: numpy array with the Harris response at each pixel.
        Output:
          - destImage: Boolean numpy array of the same shape, where True indicates 
            the pixel is the local maximum within a 7x7 neighborhood.
        Parameter hints:
          - Use ndimage.maximum_filter to get the maximum value in a 7x7 window.
          - Compare the original harrisImage with the filtered image.
        """
        # TODO: Implement local maximum detection.
        filterd_image = ndimage.maximum_filter(harrisImage, size=7, mode="constant",cval=255)
        
        height, width = harrisImage.shape
        destImage = np.zeros((height,width), dtype=bool)

        for i in range(height):
            for j in range(width):
                if filterd_image[i][j] == harrisImage[i][j]:
                    destImage[i][j]=True
                else:
                    destImage[i][j]=False
        return destImage

    def detectKeypoints(self, image):
        """
        Input:
          - image: uint8 BGR image with pixel values in [0, 255].
        Output:
          - A list of cv2.KeyPoint objects, each with:
              • pt: (x, y) coordinates,
              • angle: gradient orientation (in degrees) at that point,
              • response: Harris response at that point,
              • size: fixed to 10.
        Parameter hints:
          - Convert image to float32 and normalize to [0,1], then convert to grayscale.
          - Call computeHarrisValues to obtain harrisImage and orientationImage.
          - Call computeLocalMaxima to get a boolean mask for local maxima.
          - Iterate over the image; for each pixel that is a local maximum, create a keypoint.
        """
        # TODO: Implement keypoint detection using the Harris method.
        image_float32 = image.astype(np.float32) / 255
        image_gray = cv2.cvtColor(image_float32,cv2.COLOR_BGR2GRAY) 
        harris,grad = self.computeHarrisValues(image_gray)
        maxima = self.computeLocalMaxima(harris)

        height,width = image_gray.shape

        threshold = 0.08
        keypoints = []
        for i in range(height):
            for j in range(width):
              if maxima[i][j] == True and harris[i][j] > harris.max() * threshold:
                  temp_keypoint = cv2.KeyPoint(x=j, y=i, angle=grad[i][j], response=harris[i][j], size=10)
                  keypoints.append(temp_keypoint)
        return keypoints
                  
class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        """
        Input:
          - image: uint8 BGR image.
        Output:
          - A list of keypoints detected using OpenCV's ORB.
        """
        detector = cv2.ORB_create()
        return detector.detect(image)


## Feature descriptors #########################################################

class FeatureDescriptor(object):
    def describeFeatures(self, image, keypoints):
        """
        Input:
          - image: uint8 BGR image.
          - keypoints: list of detected keypoints.
        Output:
          - A numpy array of descriptors with shape: (number of keypoints, descriptor dimension).
        """


class SimpleFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Input:
          - image: uint8 BGR image.
          - keypoints: list of keypoints.
        Output:
          - desc: A (K, 25) numpy array where each descriptor is a flattened 5x5 intensity window.
        Parameter hints:
          - Convert the image to float32, normalize to [0, 1], and convert to grayscale.
          - For each keypoint (x, y), extract a 5x5 patch centered at (x, y).
          - If the patch goes beyond the image borders, fill those areas with zeros.
        """        
        image_float32 = image.astype(np.float32) / 255
        image_gray = cv2.cvtColor(image_float32,cv2.COLOR_BGR2GRAY)
        key_len = len(keypoints)
        height, width = image_gray.shape

        descriptors = np.zeros((key_len,25))
        for i in range(key_len):
          x, y = keypoints[i].pt # get the col and row of the keypoints
          for j in range(25):
            if y + (j / 5 - 2) >= 0 and y + (j / 5 - 2) < height and x + (j % 5 - 2) >=0 and x + (j % 5 - 2) < width:
              descriptors[i][j] = image_gray[int(y + (j / 5 - 2))][int(x + (j % 5 - 2))]

        return descriptors


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Input:
          - image: uint8 BGR image.
          - keypoints: list of keypoints.
        Output:
          - desc: A (K, windowSize^2) numpy array, e.g., with windowSize = 8.
        Parameter hints:
          - Normalize image and convert to grayscale, then apply Gaussian filtering.
          - For each keypoint, compute an affine transformation that maps a 40x40 window
            around the keypoint to an 8x8 window based on its position and orientation.
          - Use cv2.warpAffine to sample the transformed window.
          - Normalize the resulting descriptor (zero mean, unit variance; if variance is too small, set descriptor to zero).

        问题在于，cv2.warpAffine只接受2*3的矩阵，这和正常的矩阵是不一样的，所以我准备加一个函数来去掉最后一行，而transformation里生成的则是3*3
        的矩阵，以方便我们进行矩阵运算
        这里还有一个问题，在旋转矩阵的时候PPT里的方法是R[x,y],但warpaffine里面其实是[x,y]R，所以旋转矩阵是转置
        最后，实现**绕一点旋转**的方法就是平移*旋转*平移的逆矩阵
        """
        # TODO: Implement the MOPS feature descriptor.
        image_float32 = image.astype(np.float32) / 255
        image_gray = cv2.cvtColor(image_float32,cv2.COLOR_BGR2GRAY)
        image_gray_gaussian = cv2.GaussianBlur(image_gray,ksize=(5,5),sigmaX=1.5)

        windowSize = 8
        key_len = len(keypoints)
        height, width = image_gray.shape

        desc = np.zeros((key_len,windowSize*windowSize))

        for i in range(key_len):
          image_gray_gaussian_dup = np.copy(image_gray_gaussian)
          angle = -1 * keypoints[i].angle
          x, y = keypoints[i].pt
          angle_in_radian = math.radians(angle)

          s_matrix = transformations.get_scale_mx(0.2,0.2)
          translation = transformations.get_trans_mx(np.array([-x,-y]))
          r_matrix = transformations.get_rot_mx(angle_in_radian)
          translation_inv = np.linalg.inv(translation)
          image_gray_cv = np.copy(image_gray_gaussian_dup)

          r_cv = cv2.getRotationMatrix2D((x,y),angle,scale=0.2)

          # 这个是正确的旋转+scale
          final_matrix = np.dot(np.dot(np.dot(translation_inv,s_matrix),r_matrix),translation)

          final_matrix = transformations.delete_last_row(final_matrix)

          image_affined = cv2.warpAffine(image_gray_gaussian_dup,final_matrix,dsize=(width,height))
          # 中心在左上角
          for j in range(64):
            if y + (j / 8 - 3) >= 0 and y + (j / 8 - 3) < height and x + (j % 8 - 3) >=0 and x + (j % 8 - 3) < width:
              desc[i, j] = image_affined[int(y + (j / 8 - 3))][int(x + (j % 8 - 3))]
            
          mean = np.mean(desc[i,:])
          std = np.std(desc[i,:])
          if std < 1e-10:
              desc[i,:]=0
          else:
              desc[i,:] = (desc[i,:] - mean) / std

        return desc #,image_affined

class ORBFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Use OpenCV's ORB to compute descriptors.
        
        Input:
          - image: uint8 BGR image.
          - keypoints: list of keypoints.
        Output:
          - A numpy array of descriptors with shape (K, 128).
        """
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))
        return desc


class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        """
        Input:
          - image: uint8 BGR image.
          - keypoints: list of keypoints.
        Output:
          - A custom descriptor numpy array with shape (K, descriptor_dimension).
        Hint:
          - You may combine ideas from SimpleFeatureDescriptor and MOPSFeatureDescriptor.
        """
        
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################

class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
          - desc1: numpy array of shape (n, d) representing descriptors for image 1.
          - desc2: numpy array of shape (m, d) representing descriptors for image 2.
        Output:
          - A list of cv2.DMatch objects. For each match, set:
              • queryIdx: index in desc1,
              • trainIdx: index in desc2,
              • distance: distance between descriptors.
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        matches = bf.match(desc1,desc2)

        return matches

    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        """
        Evaluate matches using a ground-truth homography.
        Input:
          - features1: keypoints from image 1.
          - features2: keypoints from image 2.
          - matches: list of cv2.DMatch objects.
          - h: homography matrix (array of 9 elements).
        Output:
          - Average SSD distance between transformed and actual keypoint positions.
        """
        d = 0
        n = 0
        for m in matches:
            id1, id2 = m.queryIdx, m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)
            d += np.linalg.norm(ptNew - ptOld)
            n += 1
        return d / n if n != 0 else 0

    @staticmethod
    def applyHomography(pt, h):
        """
        Transform a point using a homography.
        Input:
          - pt: (x, y) tuple.
          - h: homography vector (length 9).
        Output:
          - Transformed (x, y) as a numpy array.
        """
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]
        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
                         (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
          - desc1: numpy array with shape (n, d).
          - desc2: numpy array with shape (m, d).
        Output:
          - A list of cv2.DMatch objects using SSD (Euclidean distance) for nearest-neighbor matching.
        Parameter hints:
          - Use scipy.spatial.distance.cdist to compute the distance matrix. 在前面的参数作为竖向的坐标
          - For each descriptor in desc1, find the closest descriptor in desc2.
        """
        distances = spatial.distance.cdist(desc1,desc2,'euclidean')

        indices = np.argmax(distances,axis=1)
        
        len_1 = desc1.shape[0]
        matches=[]

        for i in range(len_1):
          matches.append(cv2.DMatch(i,indices[i],distances[i,indices[i]]))
          
        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        """
        Input:
          - desc1: numpy array with shape (n, d).
          - desc2: numpy array with shape (m, d).
        Output:
          - A list of cv2.DMatch objects using the ratio test.
        Parameter hints:
          - For each descriptor in desc1, compute distances to all descriptors in desc2.
          - Identify the two nearest neighbors and compute the distance ratio.
          - Use the ratio as the matching score.
        """
        distances = spatial.distance.cdist(desc1,desc2,'euclidean')
        indices = np.argmax(distances,axis=1)

        amount = desc1.shape[0]
        distances_sorted = np.sort(distances)
        two_nearest = distances_sorted[:,-2:]
        matches = []

        ratio_distance = np.where(two_nearest[:,0] != 0,two_nearest[:,1] / two_nearest[:,0], 0)

        for i in range(amount):
            if ratio_distance[i] < 0.9:
                matches.append(cv2.DMatch(i,indices[i],distances[i,indices[i]]))
        '''
        distances = spatial.distance.cdist(desc1,desc2,'euclidean')

        indices = np.argmax(distances,axis=1)
        
        len_1 = desc1.shape[0]
        matches=[]
        for i in range(len_1):
          matches.append(cv2.DMatch(i,indices[i],distances[i,indices[i]]))
          
        return matches
        '''
        return ratio_distance


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
