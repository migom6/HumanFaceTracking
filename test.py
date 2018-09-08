# from detec import FaceDetection
import cv2
import numpy as np
path1 = '/Users/migom/Desktop/summer/haar/new/facerecognition/test-images/2.png'
path2 = '/Users/migom/Desktop/summer/haar/new/facerecognition/test-images/3.jpg'
img1 = cv2.imread(path1, 1)
img2 = cv2.imread(path2, 1)


from anchor import getAnchor
raghav = getAnchor('raghav')
anchor_faces = raghav.getfaces('test-images/')

# myModel = FaceDetection(anchor_faces)
for idx, i in enumerate(anchor_faces):
  i = np.array(i)
  cv2.imwrite('we'+str(idx)+'.png', i)
# myModel = FaceDetection(img1)
# # print(myModel.cv_predict_mat(img2))
# print(myModel.cv_predict(img2))


