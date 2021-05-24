import Tools as tl
import numpy as np
from scipy.io import loadmat
import sys
from scipy import stats, optimize, interpolate
from Cluster import Cluster, ClusterNode
from anytree import Node, RenderTree
import yaml
from anytree.exporter import DictExporter


# np.set_printoptions(threshold=sys.maxsize)


contourdatamat = loadmat('../contourdata_M_M.mat')

contourdatamat = contourdatamat['contourdata'][0][0]

contourdata = {}

contourdata['X'] = contourdatamat[0]
contourdata['Y'] = contourdatamat[1]
contourdata['File'] = contourdatamat[2]
contourdata['fl'] = contourdatamat[3]
contourdata['SectionsID'] = contourdatamat[4]
contourdata['Frames'] = contourdatamat[5]
contourdata['VideoFrames'] = contourdatamat[6]
contourdata['mean_vtshape'] = contourdatamat[7]
contourdata['U_gfa'] = contourdatamat[8]
contourdata['weights'] = contourdatamat[9]
contourdata['Xsim'] = contourdatamat[10]
contourdata['Ysim'] = contourdatamat[11]
contourdata['tv'] = contourdatamat[12]
contourdata['tvsim'] = contourdatamat[13]
contourdata['centers'] = contourdatamat[14]
contourdata['fwd'] = contourdatamat[15]
contourdata['jac'] = contourdatamat[16]
contourdata['jacDot'] = contourdatamat[17]
contourdata['Nobs'] = contourdatamat[18]
contourdata['crit'] = contourdatamat[19]
contourdata['minSize'] = contourdatamat[20]
contourdata['nClusters'] = contourdatamat[21]
contourdata['linear'] = contourdatamat[22]
contourdata['nLinClusters'] = contourdatamat[23]
contourdata['parameters'] = contourdatamat[24] # normalized weights w as in line 75
contourdata['strategies'] = contourdatamat[25]

frameRate = 83.333333 # 12 millis per frame

verbose = 1
crit = 0.1

nf = 8
nz = 6

tvsim = contourdata['tvsim'][0]
# print(len(tvsim))

nObs = len(tvsim[0][0][0][0])
# print(nObs)

z = np.empty((nObs, nz))
z[:] = np.NaN

for j in range(nz):
    # x = tvsim[j][0][0][2]
    # print(x.shape)
    for i in tvsim[j][0][0]:
        if i.shape[1] == 1:
            z[:, j:j+1] = i
    # z[:, j:j+1] = tvsim[j][0][0][0]

print(z)

w = stats.zscore(contourdata['weights'][:, 0:nf])

print(w)

dzdt, dwdt = tl.getGrad(z, w, frameRate, contourdata['File'])

# print(dzdt)
# print(dwdt)

Nobs = z.shape[0]
# lib = {true(Nobs,1)};       % library has one cluster for whole data-set

# TODO: modify lib, centers, fwd, jac, jacDot to be numpy arrays instead of python lists
lib = np.array([np.ones(Nobs, dtype=bool)])
# lib = [np.ones(Nobs, dtype=bool)]
# centers = zeros(0);         % center container
# centers = np.zeros(0)
centers = []
# fwd = cell(0);              % forward map container
fwd = []
# jac = cell(0);              % jacobian container
jac = []
# jacDot = cell(0);           % time-derivative of jacobian container
jacDot = []
# clusterInd = zeros(Nobs,1); % cluster membership indicator

# clusterInd = np.zeros((Nobs, 1))
clusterInd = np.zeros(Nobs)
"""May have to change^^^^^^
"""
# linInd = zeros(0);          % cluster linearity indicator
linInd = np.zeros(0)

# % clustering parameters
# minSize = size(z,2)+1      #% minimum cluster size (no. elements)
minSize = z.shape[1] + 1
k = 2                      #% clusters break in two

# print(Nobs)
# print(lib)
# print(centers)
# print(fwd)
# print(jac)
# print(jacDot)
# print(clusterInd[0])
# print(linInd)
# print(minSize)

parents = np.array([None])
# parents = [None]
ident = 0
clusterNode = None
head = None
nonlinear = []

# while len(lib) > 0:
while lib.shape[0] > 0:
    curCluster = lib[0]
    curParent = parents[0]

    # rem = cellfun( @ (x) all(x == curCluster), lib);
    # lib = lib(~rem);
    """No idea what this ^^ funny stuff is doing but from stepping thru debugging
    Seems like its just popping the top off
    """
    # lib.pop(0)
    # parents.pop(0)
    lib = lib[1:]
    parents = parents[1:]

    # print(z.shape)
    # print(w.shape)
    # print(curCluster.shape)
    linear, dzdw, resid = tl.linearityTest(z[curCluster == 1,:], w[curCluster == 1, :], crit)

    # print(linear)
    # print(dzdw)
    # print(resid)

    # if linear
    #     % add center and jac to containers
    #     [centers,fwd,jac,jacDot,clusterInd,linInd] = addCluster(curCluster,dzdt,dwdt,z,w,dzdw,resid,centers,fwd,jac,jacDot,clusterInd,linInd,linear,verbose);
    # else
    #     % break cluster into k smaller clusters
    #     [lib,centers,fwd,jac,jacDot,clusterInd,linInd] = breakCluster(curCluster,lib,dzdt,dwdt,z,w,k,minSize,centers,fwd,jac,jacDot,clusterInd,dzdw,resid,linInd,linear,verbose);
    # end
    # print(linear)
    if linear:
        # centers, fwd, jac, jacDot, clusterInd, linInd = tl.addCluster(curCluster, dzdt, dwdt, z, w, dzdw, resid, centers, fwd, jac, jacDot, clusterInd, linInd, linear, verbose)
        clusterNode, clusterInd = tl.addCluster(curCluster, dzdt, dwdt, z, w, dzdw, resid, clusterInd, linear, verbose, curParent, ident, nonlinear)
                                        #      curCluster, dzdt, dwdt, z, w, dzdw, resid, clusterInd, linear, verbose, curParent, ident):
    else:
        lib, parents, clusterNode, clusterInd = tl.breakCluster(curCluster, lib, parents, dzdt, dwdt, z, w, k, minSize, clusterInd, dzdw, resid, linear, verbose, curParent, ident, nonlinear)
    ident += 1

    if head is None:
        head = clusterNode

    print()
    print()

print(RenderTree(head))

dct = DictExporter().export(head)
print(yaml.dump(dct, default_flow_style=False))
with open("../clusterNode_M_M.yaml", "w") as file:  # doctest: +SKIP
    yaml.dump(dct, file)



    # break

print()

print()

print("No. observed data-points: ")
print(Nobs)
print("Linearity criterion: ")
print(crit)
print("Minimum cluster size: ")
print(minSize)
print("No. clusters: ")
print(len(nonlinear))
print("No. clusters per observed data-point: ")
print(len(nonlinear)/Nobs)
print("Percent of clusters which are non linear: ")
print(100 * sum(nonlinear) / len(nonlinear))
# print(linInd)

# parent object has only center
# either children is either leaf or non leaf
# either class will have the necesary information
# Look into pickle or dill to storing information

