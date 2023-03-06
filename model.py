'''
Temporally Smooth Sparse Coding of Grayscale Movies
'''

import cv2 as cv
import numpy as np
from dataloader import PatchGenerator
import utils.helperFunctions as hf
import utils.plotFunctions as pf
import matplotlib.pyplot as plt
from sklearn.decomposition import SparseCoder
from tqdm import tqdm

class SparseCoding():
    samples = 80

    # MODEL PARAMS
    dict_size = 500
    input_pixels = 320
    activation_threshold = 0.95
    lambdav = 0.7
    batch_size = 81
    lrn_rate = 0.01
    num_trials = 1
    eta = 16e-4

    # INFERENCE PARAMS
    tau = 20
    inference_steps = 400

    def __init__(self,patches):
        self.patches = patches
        pass

    
    def threshold_activations(self,u,alpha=0.9,gamma=100):
        '''
        Compute the activity of the neurons using the membrane potentials (u) using soft thresholding

        a = T(u) = u - threshold, u > threshold
                   u + threshold, u < -threshold
                   0, otherwise
        '''
        a = (np.abs(u) - alpha*self.activation_threshold)/(1+np.exp(-gamma*(np.abs(u)-self.activation_threshold)))
        # a = u - self.activation_threshold
        a[np.where(a<0)] = 0
        a = np.sign(u) * a
        return a

    def inference(self,batch_samples,phi,u_prev):
        b = phi.T @ batch_samples
        gramian = phi.T @ phi - np.identity(int(phi.shape[1]))
        u = u_prev
        # for step in range(self.inference_steps):
        a = self.threshold_activations(u)
        du = b - u - gramian @ a
        # (144,81) - 
        u += (1.0 / self.tau) * du
        return u, self.threshold_activations(u)

    def update_weights(self, dictionary, batch_samples, activities,lrn_rate):
        reconstructed_samples = dictionary @ activities
        # print(batch_samples)
        reconstruction_error = batch_samples - reconstructed_samples
        # print(np.mean(reconstruction_error))
        d_dictionary = reconstruction_error @ activities.T 
        dictionary = dictionary + lrn_rate * d_dictionary
        return (dictionary, reconstruction_error)
    
    def train(self, temp_smooth):
        # initialize dictionary with random values
        dictionary = hf.l2Norm(np.random.randn(self.input_pixels,self.dict_size))
        # print('dict_size:',dictionary.shape)
        learn_rate = self.eta / self.batch_size

        previous_potentials = np.random.uniform(-1,1,(self.dict_size,self.batch_size))
        activities_previous = self.threshold_activations(previous_potentials)
        sparsity_measures = []
        reconstruction_accs = []
        MSE = []
        # print(self.patches.shape)
        final_activities = np.zeros((9,self.dict_size,9))
        # fig,ax = plt.subplots(nrows=8,ncols=8)
        for image_num in tqdm(range(len(self.patches))):
            # print(image_num)
            img_patches = self.patches[image_num]
            # mean = np.mean(img_patches)
            # std = np.std(img_patches)
            # img_patches = (img_patches-mean)/std
            # img_patches = 1/(1+np.exp(-self.patches[image_num]))

            # fig,ax = plt.subplots(nrows=1,ncols=9)
            
            # print(activities_previous.shape)
            # np.random.permutation(len(img_patches[0])//self.batch_size)
            for k in np.random.permutation(len(img_patches)//self.batch_size):
            # for k in range(len(img_patches)//self.batch_size):
                # print(k*self.batch_size,(k+1)*self.batch_size)
                batch_patches = img_patches[k*self.batch_size:(k+1)*self.batch_size].T
                # print(k*self.batch_size)
                # for i in range(9):
                #     ax[i].imshow(batch_patches[:,i].reshape(16,20))
                # plt.show()
                
                # print(batch_patches.shape)
                # print(img_patches.shape)
                if image_num == 0:
                    previous_potentials = np.zeros((self.dict_size,self.batch_size))

                random_potentials = np.random.uniform(-1,1,previous_potentials.shape)
                for trial in range(self.num_trials):
                    if temp_smooth:
                        if trial == (self.num_trials-1):
                            previous_potentials, activities =  self.inference(batch_patches,dictionary,previous_potentials)
                        else:
                            _, activities =  self.inference(batch_patches,dictionary,previous_potentials)

                    else:
                        if trial == (self.num_trials-1):
                            previous_potentials, activities =  self.inference(batch_patches,dictionary,random_potentials)
                        else:
                            _, activities =  self.inference(batch_patches,dictionary,random_potentials)

                y_hist = abs(activities_previous-activities).flatten()
                y_hist = -np.sort(-y_hist)
                sparsity = 1-(np.count_nonzero(y_hist)/len(y_hist))
                sparsity_measures.append(sparsity)
                dictionary = hf.l2Norm(dictionary)
                # if image_num == 78:
                #     print(image_num)
                    # final_activities[k] = activities
                # print(np.argsort(np.mean()))
                # print(np.mean(dictionary[:,0]))
                (dictionary, reconstruction_err) = self.update_weights(dictionary,batch_patches,activities,learn_rate)

                # if image_num == 70:
                #     high_std = dictionary.T
                #     high_std = high_std[np.argsort(np.std(high_std,axis=1))[::-1]][:144]
                #     for i in range(64):
                #         element = high_std[i].reshape(16,20)
                #         ax[i%8,i//8].imshow(element,vmax=1)
                #         ax[i%8,i//8].axis('off')
                    
                #     fig.canvas.draw()
                #     fig.canvas.flush_events()
                #     plt.pause(0.05)
                #     plt.pause(0.001)
                # print(reconstruction_err)
                # with open('./dictionaries/dict_at_t%s.npy'%str(image_num).zfill(3),'wb') as f:
                    
                    # np.save(f,dictionary,allow_pickle=True)
                reconstruction_accs.append(1-np.mean(abs(reconstruction_err)))
            # coder = SparseCoder(dictionary.T)
            # codes = coder.transform(img_patches)
            # recon_patches = (dictionary@codes.T).T
            # MSE.append(np.sum((recon_patches-img_patches)**2)/(25920))
            # activities_previous=activities
        # print(reconstruction_accs,)
        return hf.l2Norm(dictionary), final_activities,np.array(sparsity_measures),np.array(reconstruction_accs), img_patches

datagen = PatchGenerator('eli_walk.avi',patch_size=(16,20),overlap=0)
datagen2 = PatchGenerator('moshe_walk.avi',patch_size=(16,20),overlap=0)
patches = np.concatenate([datagen.patches,datagen2.patches],axis=0)
sparse_coder = SparseCoding(patches)
dictionary, activities, sparse, recon, img_patches= sparse_coder.train(temp_smooth=True)
# recon_patches = np.zeros((81,320))
# recon_samples = dictionary@activities
coder = SparseCoder(dictionary.T)
# for i in range(81):
codes = coder.transform(datagen.patches[30])
print(np.count_nonzero(codes)/len(codes.flatten()))
print(codes.shape)
recon_patches = (dictionary@codes.T).T
print(np.sum((recon_patches-datagen2.patches[78])**2))
# coder = SparseCoder(dictionary.T)
# code = 
plt.figure()
plt.title('Model performance')
plt.plot(recon,label='reconstruction accuracy')
plt.plot(sparse,label='sparsity')
plt.yticks(np.arange(0,1,0.1))
plt.ylim([0,1])
plt.grid(axis='y')

plt.legend()
plt.tight_layout()
plt.show()
# plt.figure()
reshaped_patches =np.zeros((81,16,20))
for i in range(81):
    reshaped_patches[i] = recon_patches[i].reshape(16,20)
reshaped_patches = reshaped_patches.reshape((9,9,16,20),order='F')
new_img = np.zeros((144,180))
for i in range(9):
    row = np.hstack([*reshaped_patches[i]])
    new_img[i*16:(i+1)*16] = row
# new_img = np.vstack(new_img)
# new_img = np.hstack(new_img)
# fig,ax = plt.subplots(nrows=9,ncols=9,figsize=(6,5))

reshaped_patches =np.zeros((81,16,20))
for i in range(81):
    reshaped_patches[i] = datagen.patches[30,i].reshape(16,20)
reshaped_patches = reshaped_patches.reshape((9,9,16,20),order='F')
original_img = np.zeros((144,180))
for i in range(9):
    row = np.hstack([*reshaped_patches[i]])
    original_img[i*16:(i+1)*16] = row
# for i in range(81):
#     # print(np.max(datagen2.patches[78]))
#     ax[i%9,i//9].imshow(datagen2.patches[78,i].reshape(16,20),cmap='gray')
#     ax[i%9,i//9].axis('off')
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))

ax[0].imshow(original_img,cmap='gray')
ax[1].imshow(new_img,cmap='gray')
ax[0].axis('off')
ax[1].axis('off')
plt.tight_layout()
plt.show()
# # fig,ax = plt.subplots(nrows=9,ncols=9,figsize=(6,5))
# fig,ax = plt.subplots(ncols=20,nrows=10,figsize=(12,5))
# for i in range(200):
#     ax[i//20,i%20].imshow(dictionary[:,i].reshape(16,20),cmap='gray')
#     ax[i//20,i%20].axis('off')
# plt.tight_layout()
# plt.show()
plt.figure()
plt.bar(np.arange(len(codes[0])),sorted(codes[0])[::-1])
plt.show()