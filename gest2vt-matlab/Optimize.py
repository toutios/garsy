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
from scipy.optimize import minimize
import os


def main(word_filename: str = "../reference/Words/Base/11_put_gestures.txt") -> None:
    print(f'Current file: {os.path.dirname(os.path.realpath(__file__))}')
    print(f'Current Working Directory:{os.getcwd()}')
    print(f'Does file {word_filename} exist {os.path.exists(word_filename)}')
    print(f'Changing working director to file directory {os.path.dirname(os.path.realpath(__file__))}')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print(f'Does file {word_filename} exist {os.path.exists(word_filename)}')

    np.set_printoptions(threshold=sys.maxsize)

    # contourdata = loadmat('gest2vt-matlab/contourdata.mat')

    tv_a = [5.0, 7.1, 7.9, 4.9, 2.4, 0.2]
    tv_schwa = [2.6, 3.1, 5.9, 5.7, 5.6, 1.2]

    word = Word()
    word.read_gesture_file(word_filename)

    N = word.length

    targets = np.zeros((N, 6))
    omegas = np.zeros((N, 6))

    # Testing
    #
    # print(tl.get_stiffness(100))
    # print(tl.get_stiffness(150))
    # print(tl.get_stiffness(200))
    # print(tl.get_stiffness(300))
    # print(tl.get_stiffness(650))



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

    start = np.empty(0)
    end = np.empty(0)
    degree = np.empty(0)
    stiffness = np.empty(0)
    all_values = np.empty(0)
    start_end = np.empty(0)

    for i in word.get_gestures():
        # print(i.crit)

        tl.add_gesture(targets, omegas, int(i.start_s * 1000), int(i.end_s * 1000), i.mouth_part, i.degree, i.stiffness)
        # start
        # start.append(i.start_ms)
        # end.append(i.end_ms)
        # mouth.append(i.mouth_part)
        # degree.append(i.degree)
        start = np.append(start, i.start_s)
        end = np.append(end, i.end_s)
        degree = np.append(degree, i.degree)
        stiffness = np.append(stiffness, i.stiffness)
        all_values = np.append(all_values, [i.start_s, i.end_s, i.degree, i.stiffness], axis=0)
        start_end = np.append(start_end, [i.start_s, i.end_s], axis=0)

    print(all_values)


    contourdatamat = loadmat('../contourdata.mat')
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
    contourdata['parameters'] = contourdatamat[24]
    contourdata['strategies'] = contourdatamat[25]


    dct = None

    with open("../reference/clusterNode.yaml", 'r') as file:
        dct = yaml.load(file)


    head = DictImporter().import_(dct)

    # This is the actual parameter data
    param_data = contourdata['parameters'][word.index_word[0]:word.index_word[1], 0:8]


    inipars = contourdata['parameters'][word.index_word[0], 0:8]  # np.zeros((8, 1))

    phi, t = tl.simulate_vt(head, targets, omegas, inipars)


    print(param_data[:, 0])
    param_data = tl.resize_array(np.array(param_data).transpose(), N)
    # I use the below smoothed version of param_data instead of the original, which has lots of noise
    knn_param_data = tl.k_nearest_neighbors(N//20, param_data)
    print(param_data.shape)
    print(phi.shape)
    # print(param_data[0, 0:34])
    # print(phi[0, 0:34])
    squared_error = tl.squared_error(param_data, phi)

    # res = minimize(tl.calculate_error, all_values, args=(word, head, contourdata,), method='Newton-CG', jac=tl.rosen_der, hess=tl.rosen_hess, options={'disp': True})
    # res = tl.custmin_startend(tl.error_with_startend, start_end, args=(word, head, inipars, contourdata,))

    # res = tl.custmin_allvalues(tl.error_with_allvalues, all_values, args=(word, head, inipars, param_data,), stepsize=0.005, maxiter=50)
    res = tl.custmin_allvalues_w_rand(tl.error_with_allvalues, all_values, args=(word, head, inipars, knn_param_data,), stepsize=0.005, maxiter=50)
    print(res)
    print("\nUpdate the word gesture file with the formatted data below:")
    tl.pprint_allvalues(word, res['x'])


    # Below should be uncommented for Drawing

    # tl.plot_vtseq(phi, np.arange(49, N, 50), contourdata)

    figure, axis = plt.subplots(8, 2)
    for i in range(phi.shape[0]):
        # plt.plot(t, phi[i])
        # axis[i][0].plot(t, phi[i])
        # axis[i][1].plot(t, param_data[i])
        axis[i][0].plot(t, phi[i], label='Generated')
        axis[i][0].plot(t, param_data[i], label='From Observation')
        axis[i][1].plot(t, squared_error[i], label='Squared Error')

    plt.legend()
    # plt.show()

    # TODO: fix stiffness and target, optimize only start and end
    # Optimize one or two parameters at the same time

    # Open vowel is pharyngeal
    # Close front is palatal/aveolar
    # close back is velar
    # plosive is negative target
    # nasal have closed like plosive, but velopharyngeal is open
    # Fricative target is 0, not closed completely but close
    # Avoid 3, 12-16


if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        main(word_filename=args[1])
    else:
        main()
