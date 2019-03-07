import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
from pylbfgs import owlqn

import optparse

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

# default parameter options
def get_options():
    optParser = optparse.OptionParser()
    #optParser.add_option("-a", "--add-file", dest="afile", help="additional file")
    optParser.add_option("-f", dest="fileName", help="the pic that will be processed and compressed")
    # optParser.add_option("--suffix", dest="suffix",
    #                      help="suffix for output filenames")
    options, args = optParser.parse_args()
    return options

# this is the main entry point of this script
if __name__ == "__main__":
    options = get_options()

    # read original image
    Xorig = spimg.imread(options.fileName)
    ny,nx,nchan = Xorig.shape

    # # fractions of the scaled image to randomly sample at
    # sample_sizes = options.sampleSizes
    sample_sizes = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    # sample_sizes = [0.01]

    psnrList = []
    ssimList = []
    mseList = []

    # for each sample size
    Z = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
    masks = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
    for i,s in enumerate(sample_sizes):

        # create random sampling index vector
        k = round(nx * ny * s)
        ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

        # for each color channel
        for j in range(nchan):

            # extract channel
            X = Xorig[:,:,j].squeeze()

            # create images of mask (for visualization)
            Xm = 255 * np.ones(X.shape)
            Xm.T.flat[ri] = X.T.flat[ri]
            masks[i][:,:,j] = Xm

            # take random samples of image, store them in a vector b
            b = X.T.flat[ri].astype(float)

            # perform the L1 minimization in memory
            Xat2 = owlqn(nx*ny, evaluate, None, 5)

            # transform the output back into the spatial domain
            Xat = Xat2.reshape(nx, ny).T # stack columns
            Xa = idct2(Xat)
            Z[i][:,:,j] = Xa.astype('uint8')

        # saving the mask as image
        figMask = plt.figure()
        m = figMask.add_subplot(111)
        m = plt.imshow(masks[i])
        figMask.savefig(options.fileName.strip('.jpg')+'Mask'+str(s)+'.jpg')
        plt.close('all')

        # saving the reconstructed image
        figR = plt.figure()
        r = figR.add_subplot(111)
        r = plt.imshow(Z[i])
        figR.savefig(options.fileName.strip('.jpg')+'Reconstructed'+str(s)+'.jpg')
        plt.close('all')

        # psnrList.append(psnr(Xorig,Z[i], multichannel = True))
        ssimList.append(ssim(Xorig,Z[i], multichannel = True))
        # mseList.append(mse(Xorig,Z[i], multichannel = True))


    # figPSNR = plt.figure()
    # r = figPSNR.add_subplot(111)
    # r = plt.plot(psnrList)
    # figPSNR.savefig(options.fileName.strip('.jpg')+'_PSNR'+'.jpg')
    # plt.close('all')

    figSSIM = plt.figure()
    r = figSSIM.add_subplot(111)
    r = plt.plot(sample_sizes, ssimList)
    figSSIM.savefig(options.fileName.strip('.jpg')+'_SSIM'+'.jpg')
    plt.close('all')

    # figMSE = plt.figure()
    # r = figMSE.add_subplot(111)
    # r = plt.plot(mseList)
    # figMSE.savefig(options.fileName.strip('.jpg')+'_MSE'+'.jpg')
    # plt.close('all')
