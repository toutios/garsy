import numpy as np
from scipy import cluster
from sklearn.cluster import KMeans
from collections import Counter
from Cluster import Cluster, ClusterNode
from anytree import Node, RenderTree



def getGrad(z,w,t,file):
    # % getGrad - returns time derivative of task variables and factor weights,
    # % splitting by file identifier so that discontinuities are not differenced.
    # %
    # % Tanner Sorensen, March 7, 2016

    file = file[:,0]

    # fileBreaks = [1; find(diff(file)==1); size(z,1)];

    # start stop step
    # fileBreaks = np.arange(1,z.shape[0], )
    # print(z.shape)
    # print(np.diff(file))
    fileBreaks = np.array([0])

    pos = 0
    for i in np.diff(file):
        if i == 1:
            # print(pos)
            fileBreaks = np.append(fileBreaks, [pos])
        pos += 1
    fileBreaks = np.append(fileBreaks, [pos])
    # print(fileBreaks)



    # print(file)

    # print(z.shape)
    # print(w.shape)

    # z = np.empty((nObs, nz))
    # z[:] = np.NaN
    # dzdt = NaN(size(z,1),size(z,2));
    # dwdt = NaN(size(w,1),size(w,2));

    # print(z)
    # print(w)

    dzdt = np.empty((z.shape[0], z.shape[1]))
    dwdt = np.empty((w.shape[0], w.shape[1]))
    dzdt[:] = np.NaN
    dwdt[:] = np.NaN
    # print(dzdt.shape)
    # print(dwdt.shape)
    # for i=2:max(file)+1
    #     ii = fileBreaks(i-1):fileBreaks(i);
    #     [~,dzdt(ii,:)] = gradient(z(ii,:),t);
    #     [~,dwdt(ii,:)] = gradient(w(ii,:),t);
    for i in range(1, np.amax(file) + 1):
        ii = np.arange(fileBreaks[i-1], fileBreaks[i] + 1)
        # print(ii.shape)
        dzdt[ii, :] = np.gradient(z[ii, :], t)[0]
        dwdt[ii, :] = np.gradient(w[ii, :], t)[0]
        # break
    # print(dzdt)
    # print(dwdt)

    return (dzdt,dwdt)

# function [linear,dzdw,resid] = linearityTest(z,w,crit)


def linearityTest(z,w,crit):

    tmp, dzdw, resid = getFwdMap(z,w)

    # TODO: Change to making sure is under crit for all, rather than average
    resid = np.mean(np.sqrt(np.array(resid)))
    print(resid)
    linear = bool(resid < crit)
    # print(type(linear))
    # print(crit)
    return linear, dzdw, resid


def getFwdMap(z,w):

    nf = 8
    nz = 6
    int = np.ones((w.shape[0],1))
    fwd = np.zeros((nz,nf+1))

    # % LA ~ jaw + lips1 + lips2
    LzInd = 0#1
    LwInd = [0,5,6]#[1, 6, 7]
    print('W Shape: ' + str(w.shape))
    print('Z Shape: ' + str(z.shape))
    # print(w[:,LwInd].shape)
    # print(int.shape)
    # print(fwd.shape)

    # all_data = np.hstack((int, w[:,LwInd]))

    res_dL = np.linalg.lstsq(np.hstack((int, w[:,LwInd])), z[:, LzInd], rcond=None)
    dLdw = res_dL[0]
    resid1 = res_dL[1] / w.shape[0]
    # print(dLdw)
    # print(resid1)
    # fwd[LzInd: ] =

    tmp_LwInd = [0] + [i + 1 for i in LwInd]
    # [dLdw,~,resid1]=lscov([int,w(:,LwInd)],z(:,LzInd));
    # fwd(LzInd,[1 LwInd+1]) = dLdw';
    fwd[LzInd, tmp_LwInd] = np.transpose(dLdw)
    # print(fwd)

    # % TIP,PALATE,ROOT ~ jaw + tongue1 + ... + tongue4
    TRzInd = [1, 2, 4]
    TRwInd = [0, 1, 2, 3, 4]
    res_dTR = np.linalg.lstsq(np.hstack((int, w[:, TRwInd])), z[:, TRzInd], rcond=None)
    dTRdw = res_dTR[0]
    resid2 = res_dTR[1] / w.shape[0]
    # [dTRdw,~,resid2]=lscov([int,w(:,TRwInd)],z(:,TRzInd));
    tmp_TRwInd = [0] + [i + 1 for i in TRwInd]
    # fwd(TRzInd,[1 TRwInd+1]) = dTRdw';
    # print(np.transpose(dTRdw))
    # print(fwd[np.ix_(TRzInd, tmp_TRwInd)])
    fwd[np.ix_(TRzInd, tmp_TRwInd)] = np.transpose(dTRdw)
    # print(fwd)

    # % DORSUM ~ jaw + tongue1 + ... + tongue4 + vel
    DzInd = 3
    DwInd = [0, 1, 2, 3, 4, 5]
    res_dD = np.linalg.lstsq(np.hstack((int, w[:, DwInd])), z[:, DzInd], rcond=None)
    dDdw = res_dD[0]
    resid3 = res_dD[1] / w.shape[0]
    # [dDdw,~,resid3]=lscov([int,w(:,DwInd)],z(:,DzInd));
    # fwd(DzInd,[1 DwInd+1]) = dDdw';
    tmp_DwInd = [0] + [i + 1 for i in DwInd]
    fwd[DzInd, tmp_DwInd] = np.transpose(dDdw)

    # % VEL ~ vel
    VzInd = 5
    VwInd = [7]
    res_dV = np.linalg.lstsq(np.hstack((int, w[:, VwInd])), z[:, VzInd], rcond=None)
    dVdw = res_dV[0]
    resid4 = res_dV[1] / w.shape[0]
    # [dVdw,~,resid4]=lscov([int,w(:,VwInd)],z(:,VzInd));
    # fwd(VzInd,[1 VwInd+1]) = dVdw';
    tmp_VwInd = [0] + [i + 1 for i in VwInd]
    fwd[VzInd, tmp_VwInd] = np.transpose(dVdw)
    # print(fwd)

    J = fwd[:,1:]
    print(J)
    #
    resid = []
    for i in [resid1, resid2, resid3, resid4]:
        for j in i:
            resid.append(j)

    # print(resid)
    return fwd, J, resid


def getJacDot(dzdt,dwdt):

    nf = 8
    nz = 6
    jacDot = np.zeros((nz,nf))

    # % LA ~ jaw + lips1 + lips2
    LzInd = 0#1
    LwInd = [0,5,6]#[1, 6, 7]

    res_dL = np.linalg.lstsq(dwdt[:,LwInd], dzdt[:, LzInd], rcond=None)
    dLdw = res_dL[0]
    jacDot[LzInd,LwInd] = np.transpose(dLdw)

    # % TIP,PALATE,ROOT ~ jaw + tongue1 + ... + tongue4
    TRzInd = [1, 2, 4]
    TRwInd = [0, 1, 2, 3, 4]
    res_dTR = np.linalg.lstsq(dwdt[:, TRwInd], dzdt[:, TRzInd], rcond=None)
    dTRdw = res_dTR[0]
    # fwd(TRzInd,[1 TRwInd+1]) = dTRdw';
    # print(np.transpose(dTRdw))
    # print(fwd[np.ix_(TRzInd, tmp_TRwInd)])
    jacDot[np.ix_(TRzInd, TRwInd)] = np.transpose(dTRdw)
    # print(fwd)

    # % DORSUM ~ jaw + tongue1 + ... + tongue4 + vel
    DzInd = 3
    DwInd = [0, 1, 2, 3, 4, 5]
    res_dD = np.linalg.lstsq(dwdt[:, DwInd], dzdt[:, DzInd], rcond=None)
    dDdw = res_dD[0]

    jacDot[DzInd, DwInd] = np.transpose(dDdw)

    # % VEL ~ vel
    VzInd = 5
    VwInd = [7]
    res_dV = np.linalg.lstsq(dwdt[:, VwInd], dzdt[:, VzInd], rcond=None)
    dVdw = res_dV[0]
    jacDot[VzInd, VwInd] = np.transpose(dVdw)
    return jacDot


def breakCluster(curCluster, lib, parents, dzdt, dwdt, z, w, k, minSize, clusterInd, dzdw, resid, linear, verbose, curParent, ident, nonlinear):
    # % break the current cluster into k clusters
    # idx = kmeans(w(curCluster,:),k);

    # idx = cluster.vq.kmeans(w[curCluster, :], k)
    # TODO:  set random seed
    idx = KMeans(n_clusters=k, random_state=90007).fit(w[curCluster == 1,:]).labels_
    # print(idx)
    # idx = [i if i == 1 else 2 for i in idx]

    counter = Counter(idx)
    print(counter)
    # print(counter[1])
    # if sum(idx==1)>minSize && sum(idx==2)>minSize

    #     % put broken clusters in library
    #     tmp = zeros(length(curCluster),1);
    #     tmp(curCluster==1) = idx;
    #     for i=1:k
    #         lib=cat(1,lib,logical(tmp==i));
    #     end
    # else
    #     % add center, jac, and jacDot to containers
    #     [centers,fwd,jac,jacDot,clusterInd,linInd] = addCluster(curCluster,dzdt,dwdt,z,w,dzdw,resid,centers,fwd,jac,jacDot,clusterInd,linInd,linear,verbose);

    if counter[0] > minSize and counter[1] > minSize:
        center = np.mean(w[curCluster == 1, :], axis=0)
        clusterNode = ClusterNode(ident, center, None, None, None, None, False, resid, parent=curParent)

        # pass
        print('curCluster: ' + str(curCluster.shape))
        tmp = np.full((curCluster.shape[0]), -1)
        print('tmp: ' + str(tmp.shape))
        print('idx: ' + str(idx.shape))

        i = 0
        # while i < idx.shape[0]:
        #     if curCluster[i] == 1:
        #         tmp[i] = idx[i]
        #     i += 1
        # i = 0
        tmp[curCluster == 1] = idx
        for i in [0, 1]:
            arr = (tmp == i)
            # print(arr)
            lib = np.concatenate((lib, np.array([arr.astype(int)])))
            # parents = np.append(parents, clusterNode)
            # lib.append(arr.astype(int))
            parents = np.append(parents, clusterNode)
        # print(lib)

    else:
        clusterNode, clusterInd = addCluster(curCluster, dzdt, dwdt, z, w, dzdw, resid, clusterInd, linear, verbose, curParent, ident, nonlinear)

                                            #curCluster, dzdt, dwdt, z, w, dzdw, resid, clusterInd, linear, verbose, curParent, ident):

    # counter.clear()

    return lib, parents, clusterNode, clusterInd


def addCluster(curCluster, dzdt, dwdt, z, w, dzdw, resid, clusterInd, linear, verbose, curParent, ident, nonlinear):
    if linear:
        nonlinear.append(0)
    else:
        nonlinear.append(1)
    # % add center to container
    # centers = cat(1,centers,mean(w(curCluster,:),1));
    # % add forward map to container
    # fwd = cat(1,fwd,getFwdMap(z(curCluster,:),w(curCluster,:)));
    # % add jacobian to container
    # jac = cat(1,jac,dzdw);
    # % add time-derivative of jacobian to container
    # jacDot = cat(1,jacDot,getJacDot(dzdt(curCluster,:),dwdt(curCluster,:)));
    # % record data points in this cluster
    # clusterInd = clusterInd + (max(clusterInd)+1).*curCluster;
    # % record whether this cluster is truly linear
    # linInd = cat(1,linInd,linear);
    # % print
    # if verbose == true
    #     fprintf(1,'size: %d\nsqrt(MSE)=%.2f\nlinear: %d\n\n',sum(curCluster),resid,linear)

    # return centers,fwd,jac,jacDot,clusterInd,linInd
    # print(np.mean(w[curCluster == 1,:], axis=0))
    # centers.append(np.mean(w[curCluster == 1,:], axis=0))
    center = np.mean(w[curCluster == 1,:], axis=0)
    # print(getFwdMap(z[curCluster == 1, :], w[curCluster == 1, :]))
    # fwd_map_only = getFwdMap(z[curCluster == 1, :], w[curCluster == 1, :])[0]
    fwd = getFwdMap(z[curCluster == 1, :], w[curCluster == 1, :])[0]
    # print(fwd)
    # fwd.append(fwd_map_only)
    # jac.append(dzdw)
    jac = dzdw
    # print(jac)
    # print(getJacDot(dzdt[curCluster == 1, :], dwdt[curCluster == 1, :]))
    # jacDot.append(getJacDot(dzdt[curCluster == 1, :], dwdt[curCluster == 1, :]))
    jacDot = getJacDot(dzdt[curCluster == 1, :], dwdt[curCluster == 1, :])

    clusterInd = clusterInd + (np.max(clusterInd) + 1) * curCluster
    # print(clusterInd)

    # linInd = np.append(linInd, linear)
    linInd = linear

    clusterNode = ClusterNode(ident, center, fwd, jac, jacDot, linInd, True, resid, parent=curParent)
    # clusterNode.resid = resid

    return clusterNode, clusterInd

    # centers.append()

