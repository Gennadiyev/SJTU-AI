import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def getPoolingKernel(kernelSize = 25):
    sizeHalved = float(kernelSize)/2.0
    position = []
    for i in range(kernelSize):
        position.append(sizeHalved - abs(float(i)+0.5-sizeHalved))
    position = np.array(position)
    kernel = np.outer(position.T, position)
    kernel = kernel / (sizeHalved**2)
    return kernel

def getParams(patchSize, spatialBinCount):
    ks = 2*int(patchSize / (spatialBinCount+1));
    stride= patchSize // spatialBinCount
    pad = ks // 4
    return ks, stride, pad

class SIFTNet(nn.Module):
    def getCircularGaussKernel(self, kernelSize=21, circularFlag = True, sigmaType = 'hesamp'):
        halfSize = float(kernelSize) / 2.;
        r2 = float(halfSize ** 2);
        if sigmaType == 'hes':
            sigmaSquared = 0.9 * r2;
        elif sigmaType == 'sqr':
            sigmaSquared = kernelSize ** 2
        else:
            raise ValueError('Unknown sigmaType:', sigmaType)
        distance = 0;
        kernel = np.zeros((kernelSize,kernelSize))
        for y in range(kernelSize):
            for x in range(kernelSize):
                distance = (y - halfSize+0.5)**2 +  (x - halfSize+0.5)**2;
                kernel[y,x] = math.exp(-distance / sigmaSquared)
                if circularFlag and (distance >= r2):
                    kernel[y,x] = 0.
        return kernel
    def __repr__(self):
            return self.__class__.__name__ + '(' + 'angBinCount=' + str(self.angBinCount) +\
             ', ' + 'spatialBinCount=' + str(self.spatialBinCount) +\
             ', ' + 'patchSize=' + str(self.patchSize) +\
             ', ' + 'rootsift=' + str(self.rootsift) +\
             ', ' + 'sigmaType=' + str(self.sigmaType) +\
             ', ' + 'maskType=' + str(self.maskType) +\
             ', ' + 'clipval=' + str(self.clipval) + ')'
    def __init__(self,
                 patchSize = 65, 
                 angBinCount = 8,
                 spatialBinCount = 4,
                 clipval = 0.2,
                 rootsift = False,
                 maskType = 'CircularGauss',
                 sigmaType = 'hes'):
        super(SIFTNet, self).__init__()
        self.eps = 1e-10
        self.angBinCount = angBinCount 
        self.spatialBinCount = spatialBinCount
        self.clipval = clipval
        self.rootsift = rootsift
        self.maskType = maskType
        self.patchSize = patchSize
        self.sigmaType = sigmaType

        if self.maskType == 'CircularGauss':
            self.gk = torch.from_numpy(self.getCircularGaussKernel(kernl en=patchSize, circ=True, sigmaType=sigmaType).astype(np.float32))
        elif self.maskType == 'Gauss':
            self.gk = torch.from_numpy(self.getCircularGaussKernel(kernl en=patchSize, circ=False, sigmaType=sigmaType).astype(np.float32))
        elif self.maskType == 'Uniform':
            self.gk = torch.ones(patchSize,patchSize).float() / float(patchSize*patchSize)
        else:
            raise ValueError('Unknown maskType:', maskType)
            
        self.kernelSize, self.stride, self.pad = getParams(patchSize, spatialBinCount)
        self.gx = nn.Conv2d(1, 1, kernelSize=(1,3),  bias = False)
        self.gx.weight.data = torch.tensor(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        
        self.gy = nn.Conv2d(1, 1, kernelSize=(3,1),  bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))
        nw = getPoolingKernel(kernelSize = self.kernelSize)
        
        self.pk = nn.Conv2d(1, 1, kernelSize=(nw.shape[0], nw.shape[1]),
                            stride = (self.stride, self.stride),
                            padding = (self.pad , self.pad ),
                            bias = False)
        new_weights = np.array(nw.reshape((1, 1, nw.shape[0],nw.shape[1])))
        self.pk.weight.data = torch.from_numpy(new_weights.astype(np.float32))
        return

    def forward(self, x):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps)
        mag  = mag * self.gk.expand_as(mag).to(mag.device)
        oLarge = (ori + 2.0 * math.pi )/ (2.0 * math.pi) * float(self.angBinCount )
        boLarge =  torch.floor(oLarge)
        woLarge = oLarge - boLarge
        boLarge2 =  boLarge %  self.angBinCount 
        boLarge3 = (boLarge2 + 1) % self.angBinCount 
        woLarge2 = (1.0 - woLarge) * mag
        woLarge3 = woLarge * mag
        angBins = []
        for i in range(0, self.angBinCount):
            out = self.pk((boLarge2 == i).float() * woLarge2 + (boLarge3 == i).float() * woLarge3)
            angBins.append(out)
        angBins = torch.cat(angBins,1)
        angBins = angBins.view(angBins.size(0), -1)
        angBins = F.normalize(angBins, p=2)
        angBins = torch.clamp(angBins, 0., float(self.clipval))
        angBins = F.normalize(angBins, p=2)
        if self.rootsift:
            angBins = torch.sqrt(F.normalize(angBins,p=1) + 1e-10)
        return angBins