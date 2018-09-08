from detec import FaceDetection
import cv2

path1 = '/Users/migom/Desktop/summer/haar/new/face-recognition/images/Vladimir_Putin/Vladimir_Putin_0003.jpg'
path2 = '/Users/migom/Desktop/1601425_999738940065732_37126818477988456_n.jpg'
img1 = cv2.imread(path1, 1)
img2 = cv2.imread(path2, 1)


myModel = FaceDetection(img1)
# print(myModel.cv_predict_mat(img2))
print(myModel.cv_predict(img2))