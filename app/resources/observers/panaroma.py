"""
@file   panaroma.py
@author Xinyu Cai, Shanghai AI Lab
@brief  
"""

__all__ = [
    'Panaroma', 
    'draw_panoramic'
]

import os
import cv2
import numpy as np

class Panaroma:
    def image_stitch(self, images, lowe_ratio=0.75,mat=None ,max_Threshold=4.0,match_status=False):

        #detect the features and keypoints from SIFT
        (imageB, imageA) = images
        (KeypointsA, features_of_A) = self.Detect_Feature_And_KeyPoints(imageA)
        (KeypointsB, features_of_B) = self.Detect_Feature_And_KeyPoints(imageB)

        #got the valid matched points
        Values = self.matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)

        if Values is None or Values[1] is None:
            Homography = mat
        else:
            (matches, Homography, status) = Values
        Homography = mat
        result_image = self.getwarp_perspective(imageA,imageB,Homography)
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        return result_image

    def getwarp_perspective(self,imageA,imageB,Homography):
        val = imageA.shape[1] + imageB.shape[1]        
        result_image = cv2.warpPerspective(imageA, Homography, (val , imageA.shape[0]))

        return result_image
    def getwarp_perspective_inv(self,imageA,imageB,Homography):
        val = imageA.shape[1] + imageB.shape[1]
        image = np.zeros((imageB.shape[0],val,3))
        image[:,imageA.shape[1]:] = imageA

        result_image = cv2.warpPerspective(image, Homography, (val, imageB.shape[0]))
        return result_image

    def Detect_Feature_And_KeyPoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptors = cv2.SIFT_create()
        (Keypoints, features) = descriptors.detectAndCompute(image, None)

        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, features)

    def get_Allpossible_Match(self,featuresA,featuresB):

        # compute the all matches using euclidean distance and opencv provide
        #DescriptorMatcher_create() function for that
        match_instance = cv2.DescriptorMatcher_create("BruteForce")
        All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)
        return All_Matches

    def All_validmatches(self,AllMatches,lowe_ratio):
        #to get all valid matches according to lowe concept..
        valid_matches = []

        for val in AllMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))

        return valid_matches

    def Compute_Homography(self,pointsA,pointsB,max_Threshold):
        #to compute homography using points in both images

        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return (H,status)

    def matchKeypoints(self, KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):

        AllMatches = self.get_Allpossible_Match(featuresA,featuresB)
        valid_matches = self.All_validmatches(AllMatches,lowe_ratio)

        if len(valid_matches) > 4:
            # construct the two sets of points
            pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
            pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])

            (Homograpgy, status) = self.Compute_Homography(pointsA, pointsB, max_Threshold)

            return (valid_matches, Homograpgy, status)
        else:
            return None

    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]
        return (h,w)

    def get_points(self,imageA,imageB):

        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        return vis


    def draw_Matches(self, imageA, imageB, KeypointsA, KeypointsB, matches, status):

        (hA,wA) = self.get_image_dimension(imageA)
        vis = self.get_points(imageA,imageB)

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(KeypointsA[queryIdx][0]), int(KeypointsA[queryIdx][1]))
                ptB = (int(KeypointsB[trainIdx][0]) + wA, int(KeypointsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis
panaroma = Panaroma()

def draw_panoramic(images, downscale):
    
    assert len(images) == 3
    mat1 = np.array([[4.76040667e-01,-1.71954901e-02  ,1.69442996e+03],
        [-1.84314989e-01 , 9.64558843e-01  ,2.44522292e+01],
        [-3.06922807e-04 ,-3.87484989e-06  ,1.00000000e+00]],dtype=np.float32)
    mat2 = np.array([[ 3.55068453e-01  ,1.12000470e-02 , 1.70460632e+03],
        [-2.08723446e-01  ,9.67645376e-01 ,-2.00499948e+00],
        [-3.32951701e-04  ,3.57419672e-06 , 1.00000000e+00]],dtype=np.float32)
    if downscale == 2:
        mat1 = np.array([[ 4.75405766e-01, -1.03176668e-02 , 8.49076995e+02],
                        [-1.79571630e-01 , 9.73561155e-01  ,1.07936964e+01],
                        [-5.78480252e-04 , 2.08630946e-06  ,1.00000000e+00]])
        mat2 = np.array([[ 3.69706467e-01, 2.66713064e-02,  8.52135794e+02],
                        [-2.04139468e-01 ,9.77160475e-01, -2.25453340e+00],
                        [-6.59383677e-04 ,2.51398243e-05,  1.00000000e+00]])
    elif downscale == 4:
        mat1 = np.array([[4.60364813e-01, -1.66270776e-02,  4.24794837e+02],
        [-1.86072673e-01 , 9.69446566e-01 , 5.72085839e+00],
        [-1.23814247e-03 ,-2.56683431e-06 , 1.00000000e+00],])
        mat2 = np.array([[3.11609465e-01, -8.72843671e-04,  4.26591259e+02],
        [-2.26351619e-01 , 9.56561263e-01 , 7.97709960e-01],
        [-1.42074348e-03 ,-8.32899759e-06 , 1.00000000e+00],])
    else:
        # raise NotImplementedError
        pass
    result2 = panaroma.image_stitch([images[1], images[2]],mat=mat1)
    result = panaroma.image_stitch((images[0],result2),mat=mat2)
    return result



if __name__ == "__main__":
    
    basepath = '/path/to/your/image'
    subpath = r'lotd_forest_3cam_20221208_20221216124523_ds4.0'
    carmera_list = ['camera_FRONT_LEFT','camera_FRONT','camera_FRONT_RIGHT']
    filename = [os.path.join(basepath,subpath,carmera_id,'rgb_volume','00000000.png') for carmera_id in carmera_list]
    # filename.sort()
    # filename = filename[:-1]
    print(filename)
    images = []
    no_of_images = len(filename)
    for i in range(no_of_images):
        images.append(cv2.imread(filename[i]))
    res = draw_panoramic(images,4)
    cv2.imwrite("Panorama_image.jpg",res)