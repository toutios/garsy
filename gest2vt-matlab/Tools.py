import numpy as np
import math
import matplotlib.pyplot as plt
from anytree import AnyNode
from scipy.optimize import OptimizeResult
from Word import Word
from typing import *
import random

# random.seed(90007)

def getNearestCluster(w, head):

    # K-means
    # sklearn
    # scikit

    # [~,indx] = min(sqrt(sum(   (centers-  (ones(size(centers,1),1)*w(1:8)')  )  .^2,2)))

    # w is a numpy array
    if head.isLeaf:
        # print("END")
        # print()
        return head
    # otherwise recurse
    # print(np.transpose(w)[0])

    curr_center = np.transpose(w)[0]

    # print()
    # print(head.center)
    left_node = head.children[0]
    right_node = head.children[1]
    # print(len(head.children))

    left_dist = np.sqrt(np.sum(np.power(np.asarray(left_node.center) - curr_center, 2)))
    right_dist = np.sqrt(np.sum(np.power(np.asarray(right_node.center) - curr_center, 2)))
    # print(left_dist)
    # print(right_dist)
    if right_dist > left_dist:
        # print("Going to the left")
        return getNearestCluster(w, left_node)
    else:
        # print("Going to the right")
        return getNearestCluster(w, right_node)



    # mid = centers - np.ones((centers.shape[0],1)).dot(np.transpose(w))


    # print(np.argmin(np.sqrt(np.sum(np.power(mid, 2), 1))))

    # return np.argmin(np.sqrt(np.sum(np.power(mid, 2), 1)))


def jacStar(J, W, G_A, Nz):
    Jt = np.transpose(J)

    C = J.dot(np.linalg.inv(W)).dot(Jt)

    # print (C)

    Im = np.eye(Nz)

    Jstar = (np.linalg.inv(W).dot(Jt)).dot(np.linalg.inv(C + (Im - G_A)))

    # print(Jstar)

    return Jstar


def simulate_vt(head, targets, omegas, inipars):
    # nz, nphi = contourdata['jac'][0][0].shape
    nz = 6
    nphi = 8

    # print(nz)
    # print(nphi)

    omega = np.zeros((nz, 1))  # natural frequencies of task variables
    w = np.ones((nphi, 1))
    W = np.diagflat(w)

    # centers = contourdata['centers']
    # fwd = contourdata['fwd']
    # jac = contourdata['jac']
    # jacDot = contourdata['jacDot']


    zPhiRel = np.array(
        [[1, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0]], dtype=bool)

    # parameters of the neutral gesture
    # (see Saltzman & Munhall, 1989, Appendix A)
    omega_N = 10
    B_N = 2 * omega_N * np.eye(nphi)
    K_N = math.pow(omega_N,2) * np.eye(nphi)
    # G_N = np.diagflat(np.logical_not(np.transpose(zPhiRel).dot(omega))).astype(np.int64)
    G_N = np.eye(nphi)

    # print(G_N)

    n = targets.shape[0]

    phi = np.zeros((nphi, n))
    # print(inipars.shape)
    # print(phi[:, 0].shape)
    # print(phi[:, 0:1].shape)
    # VVVV This is necesary for some stupid ass reason
    # because the python interpreter interchangeable sees the inipars.shape as
    # (8,) or (8, 1) randomly
    # Find a solution that doesnt involve this shit ig
    if inipars.shape == (8, 1):
        inipars = inipars[:, 0]
    phi[:, 0] = inipars
    phi[:, 1] = inipars

    # print(phi)
    h = 0.001
    # t = h:h: (N * h)
    t = np.arange(0, (n) * h, h)
    # print(t)

    for i in range(2, n):
        # print(phi[:,:])
        #phi is this thing
        #[a, b, c, d, .... , z]
        #[a, b, c, d, .... , z]
        #[a, b, c, d, .... , z]
        #and you just getting the columns
        cluster = getNearestCluster(phi[:, i-1:i], head)
        # break

        # F = fwd[indx][0]
        # J = jac[indx][0]
        # J_t = jacDot[indx][0]
        F = np.asarray(cluster.fwd)
        J = np.asarray(cluster.jac)
        J_t = np.asarray(cluster.jacDot)


        # print(F)
        # print(w)
        # print(W)

        Jstar = jacStar(J, W, np.diag(omega), nz)

        ## Jstar = pinv(J)

        B = np.diagflat(2 * omegas[i,:])
        # print(B)
        K = np.diagflat(np.power(omegas[i,:],2))
        # print(K)

        z0 = np.transpose(targets[i,:])
        # print(z0)
        # print(z0)

        x1 = phi[:, i - 1]
        x2 = phi[:, i - 2]

        # print(x1)
        # print(x2)

        b = - G_N.dot(K_N) - Jstar.dot(K).dot(F[:, 1: 9])
        # print(b)
        a = - Jstar.dot(B).dot(J) - Jstar.dot(J_t) - B_N + Jstar.dot(J).dot(B_N) - G_N.dot(B_N)
        # print(a)
        c = Jstar.dot(K).dot(z0 - F[:, 0])

        # print(c)
        #

        x = np.linalg.inv(np.eye(8) - a * h - b * math.pow(h,2)).dot(- x2 + 2 * x1 - (a * h).dot(x1) + c * math.pow(h, 2))
        # print(x)
        phi[:, i] = x

        # break

    return phi, t


def add_gesture(targets, omegas, startms, endms, constriction, degree, stiffness):
    # print("start")
    # print(startms)
    # print("edn")
    # print(endms)
    targets[startms: endms, constriction] = degree
    # print("Stiffness: " + str(-1000 * math.log(crit) / (endms - startms + 1)))
    # Stiffness is ~45 for 100ms, ~22.5 for 200 ms, and so forth
    omegas[startms: endms, constriction] = stiffness  # -1000 * math.log(crit) / (endms - startms + 1)


def add_gesture_crit(targets, omegas, startms, endms, constriction, degree):
    targets[startms: endms + 1, constriction] = degree
    omegas[startms: endms + 1, constriction] = -1000 * math.log(0.01) / (endms - startms + 1)


def get_stiffness(length):
    return -1000 * math.log(0.01) / length


def weights_to_vtshape(weights, mean_vtshape, U_gfa):
    # return mean_vtshape + weights*pinv(U_gfa(:,0:weights.shape))
    # print(weights)
    # print(mean_vtshape + weights.dot(np.linalg.pinv(U_gfa[:, 0:weights.shape[0]])))
    return mean_vtshape + weights.dot(np.linalg.pinv(U_gfa[:, 0:weights.shape[0]]))


def plot_from_xy(xy_data, sectionID, color, curr_plot):
    # print(int(xy_data.shape[1]/2))

    # X=xy_data(1:length(xy_data)/2);
    # Y=xy_data(length(xy_data)/2+1:end);
    # print(sectionID.shape)
    sectionID = sectionID[:,0]
    X = xy_data[0, 0:int(xy_data.shape[1]/2)]
    Y = xy_data[0, int(xy_data.shape[1]/2):xy_data.shape[1]]
    # print(X.shape)
    # print(Y)
    # print(sectionID.shape)

    # X1=X(ismember(sectionID,1:6));
    # Y1=Y(ismember(sectionID,1:6));

    # X2=X(ismember(sectionID,7:10));
    # Y2=Y(ismember(sectionID,7:10));

    # X3=X(ismember(sectionID,11:15));
    # Y3=Y(ismember(sectionID,11:15));
    # X1, Y1, X2, Y2, X3, Y3 = []

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    X3 = []
    Y3 = []

    for i in range(170):
        if 1 <= sectionID[i] <= 6:
            X1.append(X[i])
            Y1.append(Y[i])
        elif 7 <= sectionID[i] <= 10:
            X2.append(X[i])
            Y2.append(Y[i])
        elif 11 <= sectionID[i] <= 15:
            X3.append(X[i])
            Y3.append(Y[i])

    # print(Y3)

    # plot(X1,Y1,color,'LineWidth',2);hold on;
    # plot(X2,Y2,color,'LineWidth',2);
    # plot(X3,Y3,color,'LineWidth',2);%hold off;

    curr_plot.plot(X1, Y1, color, "LineWidth", 2)
    curr_plot.plot(X2, Y2, color, "LineWidth", 2)
    curr_plot.plot(X3, Y3, color, "LineWidth", 2)
    # curr_plot.show()

    # axis equal;
    curr_plot.set_aspect('equal', 'box')


def plot_vtseq(phi,points,contourdata):
    # %PLOT_VTSEQ Summary of this function goes here
    # %   Detailed explanation goes here

    # figure;
    # params = phi';
    params = np.transpose(phi)
    # print(params)
    # s=std(contourdata.weights);

    s = np.std(contourdata['weights'], 0)

    # print(s)
    # for i=1:length(points)
    # print(points.shape[0])
    figure, axis = plt.subplots(1, points.shape[0])
    for i in range(points.shape[0]):
        #     subplot(1,length(points),i);
        #     plt.subplot(1, points.shape[0], i + 1)
        #     xy = weights_to_vtshape(params(points(i),:).*s(1:8), contourdata.mean_vtshape,contourdata.U_gfa);
        xy = weights_to_vtshape(params[points[i],:] * s[0:8], contourdata['mean_vtshape'], contourdata['U_gfa'])

    #     plot_from_xy(xy,contourdata.SectionsID,'k');
        plot_from_xy(xy, contourdata['SectionsID'], 'k', axis[i])
    #     axis([-35 30 -20 30]);
    #     axis off;
    #     hold off;
        axis[i].text(-30, -60, str(points[i] + 1) + ' ms')
    #     axis[i].xlim(-35, 30)
    #     axis[i].ylim(-20, 30)
        axis[i].axis('off')
        # break
    plt.show()


def resize_array(original, new_size) -> np.ndarray:
    # print(original.shape)
    # print(new_size)
    return_this = np.zeros((original.shape[0], new_size))
    for row in range(original.shape[0]):
        for col in range(new_size):
            old_spot = math.modf(col / ((new_size-1) / float(original.shape[1] - 1)))
            # print(old_spot)
            if old_spot[1] == original.shape[1] - 1:
                return_this[row][col] = original[row][original.shape[1] - 1]
                continue
            return_this[row][col] = (1 - old_spot[0]) * original[row][int(old_spot[1])] + \
                old_spot[0] * original[row][int(old_spot[1]) + 1]
    return return_this


def squared_error(generated, from_obseravation) -> float:
    return np.power(np.subtract(generated, from_obseravation), 2)


# This is with all 4 values
def error_with_allvalues(all_values, word, head, inipars, param_data) -> float:
    N = word.length
    targets = np.zeros((N, 6))
    omegas = np.zeros((N, 6))
    # inipars = np.zeros((8, 1))
    index = 0
    for i in word.get_gestures():
        start_s = all_values[index * 4]
        end_s = all_values[index * 4 + 1]
        # stiffness = all_values[index * 4 + 3]
        if start_s < 0:
            start_s = 0
        elif start_s > N / 1000:
            start_s = N / 1000
        if end_s < 0:
            end_s = 0
        elif end_s > N / 1000:
            end_s = N / 1000
        # if stiffness <= 0:
        #     stiffness = 0.00000000001
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        add_gesture(targets, omegas, start_ms, end_ms, i.mouth_part, all_values[index * 4 + 2], all_values[index * 4 + 3])
        # DONE: stiffness as a separate value from critical
        # DONE: Replace the milliseconds by seconds to normalize values
        # Overleaf
        index += 1

    phi, t = simulate_vt(head, targets, omegas, inipars)
    # TODONE: do this calculation outside
    # param_data = contourdata['parameters'][word.index_word[0]:word.index_word[1], 0:8]
    # param_data = resize_array(np.array(param_data).transpose(), N)
    error = np.sum(squared_error(param_data, phi))
    # print(error)
    # with open("../reference/error_plot.txt", 'a') as file:
    #     file.write(str(int(error)) + "\n")
    # print(error)
    # print(all_values)
    return error


def error_with_startend(start_end, word, head, inipars, param_data) -> float:
    N = word.length
    targets = np.zeros((N, 6))
    omegas = np.zeros((N, 6))
    # inipars = np.zeros((8, 1))
    index = 0
    for i in word.get_gestures():
        start_s = start_end[index * 2]
        end_s = start_end[index * 2 + 1]
        # stiffness = all_values[index * 4 + 3]
        if start_s < 0:
            start_s = 0
        elif start_s > N / 1000:
            start_s = N / 1000
        if end_s < 0:
            end_s = 0
        elif end_s > N / 1000:
            end_s = N / 1000
        # if stiffness <= 0:
        #     stiffness = 0.00000000001
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        add_gesture(targets, omegas, start_ms, end_ms, i.mouth_part, i.degree, i.stiffness)
        # DONE: stiffness as a separate value from critical
        # DONE: Replace the milliseconds by seconds to normalize values
        # Overleaf
        index += 1

    phi, t = simulate_vt(head, targets, omegas, inipars)
    # param_data = contourdata['parameters'][word.index_word[0]:word.index_word[1], 0:8]
    # param_data = resize_array(np.array(param_data).transpose(), N)
    error = np.sum(squared_error(param_data, phi))
    # print(error)
    # with open("../reference/error_plot.txt", 'a') as file:
    #     file.write(str(int(error)) + "\n")
    print(start_end)
    return error


# accept only values for start and end
def custmin_startend(fun, x0, args=(), maxfev=None, stepsize=0.001, maxiter=100, callback=None, **options):
    bestx = x0
    besty = fun(x0, *args)
    funcalls = 1
    n_iter = 0
    improved = True
    stop = False
    word = [*args][0]

    print(bestx[::2])
    print(bestx[1::2])

    # return besty
    assert np.size(bestx) % 2 == 0

    while improved and not stop and n_iter < maxiter:
        improved = False
        n_iter += 1

        for dim in range(np.size(x0)):
            for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                if 0 <= s <= word.length / 1000:
                    testx = np.copy(bestx)
                    testx[dim] = s
                    testy = fun(testx, *args)
                    funcalls += 1
                    if testy < besty:
                        print(testy)
                        # with open("../reference/error_plot.txt", 'a') as file:
                        #     file.write(str(int(testy)) + "\n")
                        besty = testy
                        bestx = testx
                        improved = True
        print(n_iter)
        if not improved:
            print("Reached End")

        # for gest_st, gest_end in zip(bestx[::2], bestx[1::2]):

        # for dim in range(np.size(x0)):
        #     for s in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
        #         testx = np.copy(bestx)
        #         testx[dim] = s
        #         testy = fun(testx, *args)
        #         funcalls += 1
        #         if testy < besty:
        #             besty = testy
        #             bestx = testx
        #             improved = True
        #     if callback is not None:
        #         callback(bestx)
        #     if maxfev is not None and funcalls >= maxfev:
        #         stop = True
        #         break

    return OptimizeResult(fun=besty, x=bestx, nit=n_iter, nfev=funcalls, success=(n_iter > 1))


# TODONE: Larger stepsizes
# Start from a large learning rate, then make it smaller
# TODO: Plot the Error as a function of Iterations
# If learning rate is too high, will jump around
# Should be smooth decreasing
def custmin_allvalues(fun, x0, args=(), maxfev=None, stepsize=0.001, maxiter=100, callback=None, **options):
    bestx = x0
    besty = fun(x0, *args)
    funcalls = 1
    n_iter = 0
    improved = True
    stop = False
    word = [*args][0]

    print(bestx[::2])
    print(bestx[1::2])
    print(bestx[2::2])
    print(bestx[3::2])

    # return besty
    assert np.size(bestx) % 4 == 0

    while improved and not stop and n_iter < maxiter:
        improved = False
        n_iter += 1

        for dim in range(np.size(x0)):
            if dim % 4 == 0 or dim % 4 == 1:
                # TODO: proportional to the duration of the gesture
                for mag in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                    if 0 <= mag <= word.length / 1000:
                        testx = np.copy(bestx)
                        testx[dim] = mag
                        testy = fun(testx, *args)
                        funcalls += 1
                        if testy < besty:
                            print(testy)
                            # print(testx)
                            # with open("../reference/error_plot.txt", 'a') as file:
                            #     file.write(str(int(testy)) + "\n")
                            besty = testy
                            bestx = testx
                            improved = True
            elif dim % 4 == 2 or dim % 4 == 3:
                # TODO: make it proportion of the value of either target or stiffness
                for mag in [bestx[dim] - stepsize*50, bestx[dim] + stepsize*50]:
                    # if 0 <= mag <= word.length / 1000:
                    testx = np.copy(bestx)
                    testx[dim] = mag
                    testy = fun(testx, *args)
                    funcalls += 1
                    if testy < besty:
                        print(testy)
                        # print(testx)
                        # with open("../reference/error_plot.txt", 'a') as file:
                        #     file.write(str(int(testy)) + "\n")
                        besty = testy
                        bestx = testx
                        improved = True

        print(n_iter)
        if not improved and stepsize > 0.001:
            print("Decrease stepsize")
            stepsize = 0.001
            improved = True
        elif not improved:
            print("Reached End")
    print("Word: " + word.word)
    return OptimizeResult(fun=besty, x=bestx, nit=n_iter, nfev=funcalls, success=(n_iter > 1))


def custmin_allvalues_w_rand(fun, x0, args=(), maxfev=None, stepsize=0.001, maxiter=100, callback=None, **options):
    bestx = x0
    besty = fun(x0, *args)
    funcalls = 1
    n_iter = 0
    improved = 0

    word = [*args][0]

    print(bestx[::2])
    print(bestx[1::2])
    print(bestx[2::2])
    print(bestx[3::2])

    # return besty
    assert np.size(bestx) % 4 == 0

    while improved < 10 and n_iter < maxiter:
        improved += 1
        n_iter += 1

        # multiple = random.random() * (math.e ** (-0.1 * n_iter + 1.5) + 10) + 0.05
        multiple = max(random.random() * 100 - 30, 0.2)
        stepsize *= multiple

        print(f'Stepsize: {stepsize}')

        for dim in range(np.size(x0)):
            if dim % 4 == 0 or dim % 4 == 1:
                # TODO: proportional to the duration of the gesture
                for mag in [bestx[dim] - stepsize, bestx[dim] + stepsize]:
                    if 0 <= mag <= word.length / 1000:
                        testx = np.copy(bestx)
                        testx[dim] = round(mag, 3)
                        testy = fun(testx, *args)
                        funcalls += 1
                        if testy < besty:
                            print(testy)
                            # print(testx)
                            # with open("../reference/error_plot.txt", 'a') as file:
                            #     file.write(str(int(testy)) + "\n")
                            besty = testy
                            bestx = testx
                            improved = 0
            elif dim % 4 == 2 or dim % 4 == 3:
                # TODO: make it proportion of the value of either target or stiffness
                for mag in [bestx[dim] - stepsize * 50, bestx[dim] + stepsize * 50]:
                    # if 0 <= mag <= word.length / 1000:
                    testx = np.copy(bestx)
                    testx[dim] = mag
                    testy = fun(testx, *args)
                    funcalls += 1
                    if testy < besty:
                        print(testy)
                        # print(testx)
                        # with open("../reference/error_plot.txt", 'a') as file:
                        #     file.write(str(int(testy)) + "\n")
                        besty = testy
                        bestx = testx
                        improved = 0

        stepsize /= multiple

        print(n_iter)
        # if not improved and stepsize > 0.001:
        #     print("Decrease stepsize")
        #     stepsize /= 5
        #     improved = True
        # elif not improved:
        #     print("Reached End")
    print("Word: " + word.word)
    return OptimizeResult(fun=besty, x=bestx, nit=n_iter, nfev=funcalls, success=(n_iter > 1))


def pprint_allvalues(word: Word, optimize_result: List[float]) -> None:
    """
    Gets the result from custmin_allvalues and prints it out in a format which you can copy
    and paste into the resulting word text files
    :param word:
    :param optimize_result:
    :return:
    """
    i = 0
    while i < len(optimize_result):
        optimize_result[i] = '{:.4f}'.format(optimize_result[i])
        i += 1
    i = 0
    for gesture in word.get_gestures():
        print(f'{optimize_result[i] * 1000}, {optimize_result[i+1] * 1000}, {gesture.mouth_part}, '
              f'{optimize_result[i+2]}, {optimize_result[i+3]}')
        i += 4


def avg_k_nearest(k: int, row: np.ndarray, col: int):
    i = col - k//2
    end = col + k//2
    sum = 0
    num_neighbors = 0
    while i < end:
        if 0 <= i < len(row):
            num_neighbors += 1
            sum += row[i]
        i += 1
    return sum / num_neighbors


def k_nearest_neighbors(k: int, two_d_array: np.ndarray) -> np.ndarray:
    """
    Takes either the param data or constriction data, then smooths things out using knn, that way it
    gets rid of some of the noisy data
    Each row is either a parameter or a constriction
    Across each row is where every value will be averaged out according to its k nearest neighbors
    :param k:
    :param two_d_array:
    :return:
    """
    arr_shape = two_d_array.shape
    return_this = np.zeros(arr_shape)
    row = 0
    while row < arr_shape[0]:
        col = 0
        while col < arr_shape[1]:
            return_this[row][col] = avg_k_nearest(k, two_d_array[row], col)
            col += 1
        row += 1
    return return_this



# Vowels have lip opening, and tongue

