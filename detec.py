from __future__ import print_function
from model import create_model
from align import AlignDlib
import numpy as np
import cv2
import chalk


class FaceDetection:
    def __init__(self, anchors, name):
        nn4_small2_pretrained = create_model()
        nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
        self.nn4_small2_pretrained =  nn4_small2_pretrained
        self.alignment = AlignDlib('models/landmarks.dat')
        anchors = [(img / 255.).astype(np.float32) for img in anchors]
        anchors_embeddings = [self.nn4_small2_pretrained.predict(np.expand_dims(anchor, axis=0))[0] for anchor in anchors]
        self.anchors_embeddings = anchors_embeddings
        self.name = name


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
                print(chalk.white('No faces'))
            return [],[[0]]
        if(verbose): print(chalk.white(str(len(faces)) + ' faces detected'))
        faces_embeddings = [self.nn4_small2_pretrained.predict(np.expand_dims(face, axis=0))[0] for face in faces]
        r = []
        for idx, f in enumerate(faces_embeddings):
            k = []
            for jdx, a in enumerate(self.anchors_embeddings):
                d = self.distance(f, a)
                if(verbose): print(chalk.yellow(d , 'face#' + str(idx) +' with anchor#'+str(jdx)))                    
                k.append(d)
            r.append(k)
        return (faces, r) 

    def cv_predict(self, frame, verbose=False):
        faces , m = self.cv_predict_mat(frame, verbose)
        avg = []
        for idx, i in enumerate(m):
            k = np.mean(i)
            avg.append(k)
            if(k < 0.56):
                print(chalk.green(self.name+" deteced"))
            k = 1
        mx = 0
        mn = 0
        if(len(m) > 0 and len(m[0]) > 0):
            mx = np.max(m)
            mn = np.min(m)                
        return faces,(mx, mn, avg)


