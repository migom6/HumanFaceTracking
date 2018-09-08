import cv2
import time
from detec import FaceDetection
from align import AlignDlib
from anchor import getAnchor
import numpy as np
   
raghav = getAnchor('raghav')
anchor_faces = raghav.getfaces('test-images/')
myModel = FaceDetection(anchor_faces, 'raghav')


frames = 0
start = time.time()
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot capture source'  
while cap.isOpened():
  ret, frame = cap.read()
  if ret:
      faces, result = myModel.cv_predict(frame, verbose=0) 
      cv2.imshow("window", frame)     
      if(len(faces)>0):
        vis = cv2.resize(faces[0], (300, 300))
        if(len(faces) > 1):
            faces = faces[1:]
            for i in faces:
                resized_image = cv2.resize(i, (300, 300)) 
                vis = np.concatenate((vis, resized_image), axis=1)
        cv2.imshow("frame", vis)
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
          break
      frames += 1
      print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
  else:
      break





