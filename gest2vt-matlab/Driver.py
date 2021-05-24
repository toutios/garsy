import numpy as np
import sys
import Tools as tl
from scipy.io import loadmat
import matplotlib.pyplot as plt
import yaml
from anytree.importer import DictImporter
from Gesture import Gesture
from Word import Word
from pprint import pprint  # just for nice printing
from anytree import RenderTree  # just for nice printing


np.set_printoptions(threshold=sys.maxsize)

# contourdata = loadmat('gest2vt-matlab/contourdata.mat')

tv_a = [5.0, 7.1, 7.9, 4.9, 2.4, 0.2]
tv_schwa = [2.6, 3.1, 5.9, 5.7, 5.6, 1.2]

word = Word()
word.read_gesture_file("../reference/Words/Paper/span_M.txt")

N = word.length

targets = np.zeros((N, 6))
omegas = np.zeros((N, 6))

inipars = np.zeros((8, 1))

# tl.add_gesture(targets, omegas, 0, 149, 1, 0.2, 0.01)
# # TT crit alveolar
#
# tl.add_gesture(targets, omegas, 50, 249, 0, -4, 0.01)
# # LIPS clo
#
# tl.add_gesture(targets, omegas, 0, 649, 4, tv_a[4], 0.01)
# # TB wide pharyngeal
#
# tl.add_gesture(targets, omegas, 450, 649, 1, -0.1, 0.01)
# # TT clo alveolar
#
# tl.add_gesture(targets, omegas, 270, 649, 5, 5, 0.01)
# # VEL wide
#
# # additional controls
#
# tl.add_gesture(targets, omegas, 250, 450, 0, tv_a[0], 0.01)
# # lips open for a
#
# tl.add_gesture(targets, omegas, 150, 449, 1, tv_a[2], 0.01)
# # TT opens
#
# tl.add_gesture(targets, omegas, 0, 269, 5, -0.1, 0.01)
# VEL closes

# word = Word()
# word.read_gesture_file("../Words/4_bet_gestures.txt")

for i in word.get_gestures():
    # print(i.crit)

    tl.add_gesture(targets, omegas, i.start_ms, i.end_ms, i.mouth_part, i.degree, i.stiffness)
    print(i)

# print(targets)
# print(omegas)


####Uncomment the two lines below to load contourdata traditionally
contourdatamat = loadmat('../contourdata_M_M.mat')
contourdatamat = contourdatamat['contourdata'][0][0]


# contourdata['contourdata'][0][0]
# gives access to the main contourdata array
# Each index reference to that array is, in sequence, referencing
# Each of the subsequent 26 arrays
# print(len(contourdatamat['contourdata'][0][0]))
# 1 contourdata.X
# 2 contourdata.Y
# 3 contourdata.File
# 4 contourdata.fl
# 5 contourdata.SectionsID
# 6 contourdata.Frames
# 7 contourdata.VideoFrames
# 8 contourdata.mean_vtshape
# 9 contourdata.U_gfa
# 10 contourdata.weights
# 11 contourdata.Xsim
# 12 contourdata.Ysim
# 13 contourdata.tv
# 14 contourdata.tvsim
# 15 contourdata.centers
# 16 contourdata.fwd
# 17 contourdata.jac
# 18 contourdata.jacDot
# 19 contourdata.Nobs
# 20 contourdata.crit
# 21 contourdata.minSize
# 22 contourdata.nClusters
# 23 contourdata.linear
# 24 contourdata.nLinClusters
# 25 contourdata.parameters
# 26 contourdata.strategies

# To make life easier for now I will create a separate contourdata object with all this info


#Uncommend the hunk of lines below to use the traditional contourdata
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
contourdata['parameters'] = contourdatamat[24]
contourdata['strategies'] = contourdatamat[25]


dct = None

with open("../clusterNode_M_M.yaml", 'r') as file:
    dct = yaml.load(file)

# also compare results by loading the tree into the normal array structure

# pprint(dct)

head = DictImporter().import_(dct)

# print(RenderTree(head))

# print(head.center)


phi, t = tl.simulate_vt(head, targets, omegas, inipars)

w_to_z = np.zeros((6, phi.shape[1]))

for i in range(phi.shape[1]):
    cluster = tl.getNearestCluster(phi[:, i], head)

    w = phi[:, i]
    fwd = np.array(cluster.fwd)[:, 1:9]
    z_c = np.array(cluster.fwd)[:, 0]

    # print(w.shape)
    # print(fwd.shape)
    # print(z_c.shape)

    z = fwd.dot(w) + z_c
    # print(w)
    # print(fwd)
    # print(z_c)
    # print(z)

    w_to_z[:, i] = z

    # break

print("GENERATED PARAMETERS")
for i in range(phi.shape[1]):
    print(" ".join(map(str, phi[:, i].tolist())))
print()

print("GENERATED CONSTRICTIONS")
for i in range(w_to_z.shape[1]):
    print(" ".join(map(str, w_to_z[:, i].tolist())))
print()

# print(phi[2])

# plot_vtseq(phi,150:100:650,contourdata)


# Below should be uncommented for Drawing

tl.plot_vtseq(phi, np.arange(149, N, 100), contourdata)

figure, axis = plt.subplots(8, 1)
for i in range(phi.shape[0]):
    # plt.plot(t, phi[i])
    axis[i].plot(t, phi[i])


plt.show()
