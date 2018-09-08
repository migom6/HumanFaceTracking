import cv2
import time
from detec import FaceDetection

path1 = '/Users/migom/Desktop/summer/haar/new/face-recognition/images/Vladimir_Putin/Vladimir_Putin_0003.jpg'
img1 = cv2.imread(path1, 1)
myModel = FaceDetection(img1)
frames = 0
start = time.time()
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot capture source'  
while cap.isOpened():
  ret, frame = cap.read()
  if ret:
      faces, _ = myModel.cv_predict(frame, verbose=1) 
      cv2.imshow("no", frame)     
      if(len(faces)>0):
        cv2.imshow("frame", faces[0])
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
          break
      frames += 1
      print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

  else:
      break





