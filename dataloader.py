import numpy as np
import pandas as pd 
import cv2 as cv
import os, sys, re
import matplotlib.pyplot as plt
from cztile.fixed_total_area_strategy import AlmostEqualBorderFixedTotalAreaStrategy2D
from cztile.tiling_strategy import Rectangle as czrect

video_dir = './walk/'

def get_avi_frames(filename):
    cap = cv.VideoCapture(os.path.join(video_dir,'moshe_walk.avi'))
    ret, frame = cap.read()
    frame = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    i=1
    frames = np.zeros((80,*frame.shape[:2]))
    
    while ret==True:
        if i>1:
            frames[i-1] = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
        else:
            frames[i-1] = frame
        ret, frame = cap.read()
        i+=1
    return frames

class PatchGenerator():
    frames = None
    patches = None
    patch_size = (4,4)
    overlap = 0

    def __init__(self,video_file,patch_size,overlap):
        self.frames = get_avi_frames(video_file)
        self.patch_size = patch_size
        self.overlap = overlap
        self.make_patches()

    def make_patches(self):
        tiler = AlmostEqualBorderFixedTotalAreaStrategy2D(total_tile_width=self.patch_size[1],
                                                            total_tile_height=self.patch_size[0],
                                                            min_border_width=self.overlap)

        tiles = tiler.tile_rectangle(czrect(x=0,y=0,w=self.frames.shape[2],h=self.frames.shape[1]))

        # for tile in tiles:
        #     print(tile.roi.x,tile.roi.y,tile.roi.w,tile.roi.h)
        
        patches = np.zeros((len(self.frames)-1,len(tiles),self.patch_size[0]*self.patch_size[1]))

        for f in range(len(self.frames)-1):
            frame = self.frames[f]
            for i in range(len(tiles)):
                tile = tiles[i]
                patches[f,i] = frame[tile.roi.y:tile.roi.y+tile.roi.h,tile.roi.x:tile.roi.x+tile.roi.w].flatten()
        patches = patches.astype(np.float32)
        patches = (patches-np.min(patches))/(np.max(patches)-np.min(patches))
        self.patches = patches
        # print(self.patches.max())
        # fig,ax = plt.subplots(nrows=36,ncols=45,figsize=(10,10))
        frame_patches = patches[78]
        # print(len(frame_patches))
        # for i in range(len(frame_patches)):
        #     ax[i//45,i%45].imshow(frame_patches[i].reshape(4,4),cmap='gray')
        #     ax[i//45,i%45].axis('off')
        # plt.tight_layout()
        # plt.show()
# datagen = PatchGenerator(video_file='moshe_walk.avi',patch_size=(4,4),overlap=0)





