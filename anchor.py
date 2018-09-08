from align import AlignDlib
import os
import cv2
import numpy as np

class getAnchor:
  def __init__(self, person):
    alignment = AlignDlib('models/landmarks.dat')
    self.alignment = alignment
    self.person = person

  def getfaces(self, path):

    def alignface(i):
      bb = self.alignment.getAllFaceBoundingBoxes(i)
      aligned_images = [self.alignment.align(96, i, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE) for bb in bb]
      return aligned_images

    count = 0
    listOfFiles = os.listdir(path)
    crop_faces = []
    for i in listOfFiles:
        i = path + i
        img = cv2.imread(i, 1)
        faces = alignface(img)
        for j in faces:
            crop_faces.append(j)
            cv2.imwrite('anchors/'+self.person+str(count)+'.jpg', j)
            count = count + 1
    return crop_faces







