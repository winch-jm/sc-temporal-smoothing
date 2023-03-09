# import ffmpeg
import os
import numpy as np
from sklearn.decomposition import SparseCoder
from dataloader import PatchGenerator
from PIL import Image
from tqdm import tqdm
dict_dir = './dictionaries'
def generate_reconstructions():
    datagen = PatchGenerator('daria_walk.avi',(8,10),overlap=0)
    frame_patches = datagen.patches[30]
    dict_files = sorted([f for f in os.listdir(dict_dir) if f.endswith('.npy')])
    for i in tqdm(range(len(dict_files))):
        phi = np.load(os.path.join(dict_dir,dict_files[i]),allow_pickle=True)
        coder = SparseCoder(phi.T)
        codes = coder.transform(frame_patches)
        # print(codes.shape)
        recon_patches = (phi @ codes.T).T
        reshaped_patches =np.zeros((324,8,10))
        for j in range(324):
            reshaped_patches[j] = recon_patches[j].reshape(8,10)
        reshaped_patches = reshaped_patches.reshape((18,18,8,10),order='F')
        new_img = np.zeros((144,180))
        for j in range(18):
            row = np.hstack([*reshaped_patches[j]])
            new_img[j*8:(j+1)*8] = row
        img = Image.fromarray((new_img*255).astype(np.uint8)).resize((900,720),resample=Image.NEAREST)
        img.save('./reconstructions/t%s.png'%str(i).zfill(3),dpi=(1080,1080))
        
generate_reconstructions()