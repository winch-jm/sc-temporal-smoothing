import matplotlib.pyplot as plt
import numpy as np


def vec2RGB(images, i=-1,dim=32):
  if i != -1:
    img = np.copy(images[i])
  else:
    img = images
  n = int(len(img) / 3)
  red = img[0:n]
  green = img[n:2 * n]
  blue = img[2 * n:3 * n]
  img = np.zeros((dim**2, 3))
  for i in range(len(red)):
    img[i, 0] = int(red[i])
    img[i, 1] = int(green[i])
    img[i, 2] = int(blue[i])
  img = np.reshape(img, (dim, dim, 3))
  return img.astype(int)

def plotFoldiak(data, title="", prevFig=None):
  (activity, thresholds, corr, latWeights) = data
  if prevFig is not None:
    (fig, rects, axisImage) = prevFig
    for rect, h in zip(rects[0], activity):
      rect.set_height(h)
    for rect, h in zip(rects[1], thresholds):
      rect.set_height(h)
    axisImage[0].set_data(corr)
    axisImage[1].set_data(latWeights)
  else:
    fig, subAxes = plt.subplots(nrows=2, ncols=2)
    rects = [None]*2
    axisImage = [None]*2
    rects[0] = subAxes[0,0].bar(np.arange(len(activity)), activity)
    subAxes[0,0].set_title("Activity")
    rects[1] = subAxes[0,1].bar(np.arange(len(thresholds)), thresholds)
    subAxes[0,1].set_ylim([0, 2])
    subAxes[0,1].set_title("Threshold")
    axisImage[0] = subAxes[1,0].imshow(corr, cmap="Greys", interpolation="nearest")
    subAxes[1,0].set_title("Output Correlations")
    subAxes[1,0].tick_params(axis="both", bottom="off", top="off", left="off", right="off")
    axisImage[1] = subAxes[1,1].imshow(latWeights, cmap="Greys", interpolation="nearest")
    subAxes[1,1].set_title("Lateral Weights")
    subAxes[1,1].tick_params(axis="both", bottom="off", top="off", left="off", right="off")
  fig.suptitle(title)
  if prevFig is not None:
    fig.canvas.draw()
  else:
    fig.show()
  return (fig, rects, axisImage)


def plotDataTiled(data, title="", prevFig=None):
  """
    Create a matplotlib plot that displays data as subplots
    Inputs:
      data [np.ndarray] 2-D array of dims (numPixels, batchSize)
        It is assumed that numPixels has an even square root
      title [str] optional string to set as the figure title
  """

  batchSize = 64
  if prevFig is not None:
    (fig, subAxes, axisImage) = prevFig
  else:
    fig, subAxes = plt.subplots(nrows=8, ncols=8)
    axisImage = [None]*len(fig.axes)
  for axisIndex, axis in enumerate(fig.axes):
    if axisIndex < batchSize:
      image = np.array(data[:,axisIndex]).astype(int)
      rgb_image =vec2RGB([image],0,8)
      if prevFig is not None:
        axisImage[axisIndex].set_data(rgb_image)
      else:
        axisImage[axisIndex] = axis.imshow(rgb_image,
          interpolation="nearest")
        axis.tick_params(
          axis="both",
          bottom="off",
          top="off",
          left="off",
          right="off")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    else:
      for spine in axis.spines.values():
        spine.set_visible(False)
      axis.tick_params(
        axis="both",
        bottom="off",
        top="off",
        left="off",
        right="off")
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
  fig.suptitle(title)
  if prevFig is not None:
    fig.canvas.draw()
    # print("none")
  else:
    # print("showing")
    fig.show()
  return (fig, subAxes, axisImage)

def plotStats(numSteps):
  fig, subAxes = plt.subplots(2, 2)
  labelList = ["Energy", "% Non-Zero Activations", "Recon Quality pSNR dB"]
  subAxes[0, 0].set_xlabel("Time Step")
  subAxes[0, 0].set_ylabel(labelList[0])
  subAxes[0, 0].set_xlim(0, numSteps)
  subAxes[0, 1].set_xlabel("Time Step")
  subAxes[0, 1].set_ylabel(labelList[1])
  subAxes[0, 1].set_xlim(0, numSteps)
  subAxes[1, 0].set_xlabel("Time Step")
  subAxes[1, 0].set_ylabel(labelList[2])
  subAxes[1, 0].set_xlim(0, numSteps)
  subAxes[1, 1].axis("off")
  fig.suptitle("Summary Statistics for LCA Sparse Coding")
  return fig, subAxes

def updateStats(figure, subAxes, i, energy, percentNonZero, SNR):
  subAxes[0, 0].scatter(i, energy, alpha=0.2, s=3, c='b')
  subAxes[0, 1].scatter(i, percentNonZero, alpha=0.2, s=3, c='b')
  subAxes[1, 0].scatter(i, SNR, alpha=0.2, s=3, c='b')
  figure.canvas.draw()


def makeSubplots2(dataList, labelList, title=""):
  assert len(dataList) == len(labelList), (
    "The lengths of dataLst and labelList must be equal.")
  numSteps = np.arange(dataList[0].size)[-1]
  fig, subAxes = plt.subplots(2, 2)
  subAxes[0, 0].scatter(np.arange(dataList[0].size), dataList[0], alpha=0.2)
  subAxes[0, 0].set_xlabel("Time Step")
  subAxes[0, 0].set_ylabel(labelList[0])
  subAxes[0, 0].set_xlim(0, numSteps)
  subAxes[0, 1].scatter(np.arange(dataList[1].size), dataList[1], alpha=0.2)
  subAxes[0, 1].set_xlabel("Time Step")
  subAxes[0, 1].set_ylabel(labelList[1])
  subAxes[0, 1].set_xlim(0, numSteps)
  subAxes[1, 0].scatter(np.arange(dataList[2].size), dataList[2], alpha=0.2)
  subAxes[1, 0].set_xlabel("Time Step")
  subAxes[1, 0].set_ylabel(labelList[2])
  subAxes[1, 0].set_xlim(0, numSteps)
  subAxes[1, 1].axis("off")
  fig.suptitle(title)
  fig.show()
