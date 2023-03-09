# Temporally smooth sparse coding


<img src="https://github.com/winch-jm/sc-temporal-smoothing/blob/master/reconstructions/original.png" width="375" height="300" title="Original Frame"></img>
<img src="https://github.com/winch-jm/sc-temporal-smoothing/blob/master/reconstructions/small_patch.gif" width="375" height="300" title="Reconstruction"></img>

Architecture based on Rozell's '08 paper on a locally competitive algorithm (LCA) for Sparse Approximation: 
https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.64.7897&rep=rep1&type=pdf

Further inspiration is taken from current neuroscience literature 

Abstract 
---------
We consider the relationship between representations of natural images in a temporally smooth sequence 
(i.e.consecutive frames in a video). Traditionally, sparse coding methods learn representations of images in isolation. 
Here, we learn an image’s sparse representation with the previous image’s representation as a starting point.

**To-Do**
---------
- [ ] Different sized dictionary elements
- [ ] Support for RGB images
- [ ] Support for Convolutional Sparse Coding
- [ ] Implement same technique in other ML architectures

Collaborators: Jeff Winchell, Dr. Edward Kim (Drexel University)
