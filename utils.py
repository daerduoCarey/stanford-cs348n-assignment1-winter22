import numpy as np
import binvox_rw
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData, PlyElement

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()
    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
        elif line.startswith('f '):
            faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))
    return np.vstack(vertices), np.vstack(faces)

def load_binvox(fn):
    with open(fn, 'rb') as fin:
        out = binvox_rw.read_as_3d_array(fin)
        return np.array(out.data)

def load_ply(fn):
    with open(fn, 'rb') as f:
        plydata = PlyData.read(f)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def viz_mesh(v, f, figsize=(4, 4)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=60)
    x = v[:, 0]
    y = v[:, 2]
    z = v[:, 1]
    for i in range(f.shape[0]):
        v1 = f[i, 0] - 1
        v2 = f[i, 1] - 1
        v3 = f[i, 2] - 1
        verts = [(x[v1], y[v1], z[v1]), \
                (x[v2], y[v2], z[v2]), \
                (x[v3], y[v3], z[v3])]
        ax.add_collection3d(Poly3DCollection(verts, edgecolor='k'))
    miv = np.min([np.min(x), np.min(y), np.min(z)])
    mav = np.max([np.max(x), np.max(y), np.max(z)])
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    plt.show()

def viz_pc(pc, figsize=(4, 4)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=60)
    x = pc[:, 0]
    y = pc[:, 2]
    z = pc[:, 1]
    ax.scatter(x, y, z, marker='.')
    miv = np.min([np.min(x), np.min(y), np.min(z)])
    mav = np.max([np.max(x), np.max(y), np.max(z)])
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    plt.show()

def viz_bv(bv, figsize=(4, 4)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=60)
    ax.voxels(bv, edgecolor='k')
    plt.tight_layout()
    plt.show()

