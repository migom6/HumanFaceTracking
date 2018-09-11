import cv2
import time
frames = 0
start = time.time()
cap = cv2.VideoCapture(0)
assert cap.isOpened(), 'Cannot capture source'  
while cap.isOpened():
  ret, frame = cap.read()
  if ret:
      
      cv2.imshow("window", frame)     
     
          
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
          break
      if key & 0xFF == ord('c'):
          cv2.imwrite('test-images/'+str(frames)+'.jpg', frame)
      frames += 1
      print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
  else:
      break