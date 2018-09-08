from model import create_model
from align import AlignDlib
import numpy as np
import cv2


class FaceDetection:
    def __init__(self, anchors):
        nn4_small2_pretrained = create_model()
        nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
        self.nn4_small2_pretrained =  nn4_small2_pretrained
        self.alignment = AlignDlib('models/landmarks.dat')
        anchors = self.import_image_from_path(anchors)
        anchors_embeddings = [self.nn4_small2_pretrained.predict(np.expand_dims(anchor, axis=0))[0] for anchor in anchors]
        self.anchors_embeddings = anchors_embeddings


    # def load_image(self, img):
    #     return img[...,::-1]

    def import_image_from_path(self, i):
        # img = self.load_image(path)
        imgs = self.align_images(i)
        imgs = [(img / 255.).astype(np.float32) for img in imgs]
        return imgs

    def align_images(self, org_img):
        bb = self.alignment.getAllFaceBoundingBoxes(org_img)
        aligned_images = [self.alignment.align(96, org_img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE) for bb in bb]
        return aligned_images

    def distance(self, emb1, emb2):
        return np.sum(np.square(emb1 - emb2))

    def cv_predict_mat(self, frame, verbose = False):
        faces = self.import_image_from_path(frame)
        if(len(faces) == 0 or isinstance(faces, int)):
            if(verbose):
                print('No faces')
            return [],[[0]]
        if(verbose): print(str(len(faces)) + ' faces detected')
        faces_embeddings = [self.nn4_small2_pretrained.predict(np.expand_dims(face, axis=0))[0] for face in faces]
        r = []
        for idx, f in enumerate(faces_embeddings):
            k = []
            for jdx, a in enumerate(self.anchors_embeddings):
                d = self.distance(f, a)
                if(verbose): print(d , 'face#' + str(idx) +' with anchor#'+str(jdx))
                k.append(d)
            r.append(k)
        return (faces, r) 

    def cv_predict(self, frame, verbose=False):
        faces , m = self.cv_predict_mat(frame, verbose)
        mx = 0
        mn = 100
        c = 0
        s = 0 
        avg = 0
        for i in m:
            for j in i:
                s = s + j
                c = c + 1
                if(j > mx):
                    mx = j
                if(j < mn):
                    mn = j
        if(c != 0 ): avg = s/c
        
        return faces,(mx, mn, avg)


