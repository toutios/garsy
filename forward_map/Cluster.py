from anytree import NodeMixin, RenderTree


class Cluster(object):  # Just an example of a base class
    # def __init__(self, center, fwd, jac, jacDot, clusterInd, linInd):
    def __init__(self, center, fwd, jac, jacDot, linInd, isLeaf, resid):
        self.center = None if center is None else center.tolist()
        self.fwd = None if fwd is None else fwd.tolist()
        self.jac = None if jac is None else jac.tolist()
        self.jacDot = None if jacDot is None else jacDot.tolist()
        # self.clusterI
        self.linInd = linInd
        self.isLeaf = isLeaf
        self.resid = None if resid is None else resid.tolist()

class ClusterNode(Cluster, NodeMixin):  # Add Node feature
    # def __init__(self, name, length, width, parent=None, children=None):
    def __init__(self, name, center, fwd, jac, jacDot, linInd, isLeaf, resid, parent=None, children=None):
        # super(Cluster, self).__init__()
        # Cluster.__init__(self, center, fwd, jac, jacDot, linInd, isLeaf)
        super().__init__(center, fwd, jac, jacDot, linInd, isLeaf, resid)
        self.name = name
        # self.length = length
        # self.width = width
        self.parent = parent
        if children:
            self.children = children

