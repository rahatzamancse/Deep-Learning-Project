import numpy as np
import plotly.graph_objs as go
import cv2
import itertools
# import open3d as o3d
import io
import os
from PIL import Image
import tensorflow as tf
import copy
import pandas as pd
import trimesh
from functools import cache
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS
import json
from functools import lru_cache

shapenet_category_to_id = {
    'airplane'	: '02691156',
    'bench'		: '02828884',
    'cabinet'	: '02933112',
    'car'		: '02958343',
    'chair'		: '03001627',
    'lamp'		: '03636649',
    'monitor'	: '03211117',
    'rifle'		: '04090263',
    'sofa'		: '04256520',
    'speaker'	: '03691459',
    'table'		: '04379243',
    'telephone'	: '04401088',
    'vessel'	: '04530566'
}

shapenet_id_to_category = dict(reversed(list(shapenet_category_to_id.items())))

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

@cache
def load_modelnet():
    return tf.keras.utils.get_file(
        "modelnet.zip",
        "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        extract=True,
    )


def modify_to_shape(voxel_grid, target_shape):
    """
    Modify the input voxel data to the target shape by padding or cropping.
    """
    # Pad or crop each dimension to match the target shape
    voxel_data = voxel_grid.matrix
    modified_data = np.zeros(target_shape, dtype=bool)

    for i in range(3):
        # If the voxel dimension is less than target, pad it
        if voxel_data.shape[i] < target_shape[i]:
            pad_amount = target_shape[i] - voxel_data.shape[i]
            # Split padding equally to both sides when possible
            pad_before = pad_amount // 2
            pad_after = pad_amount - pad_before

            padding = ((pad_before, pad_after), (0, 0), (0, 0))
            padding = padding[-i:] + padding[:-i]  # Rotate to apply to correct dimension
            voxel_data = np.pad(voxel_data, padding, mode='constant')

        # If the voxel dimension is more than target, crop it
        elif voxel_data.shape[i] > target_shape[i]:
            crop_amount = voxel_data.shape[i] - target_shape[i]
            # Split cropping equally from both sides when possible
            crop_before = crop_amount // 2
            crop_after = crop_amount - crop_before

            slicing = (slice(crop_before, -crop_after), slice(None), slice(None))
            slicing = slicing[-i:] + slicing[:-i]  # Rotate to apply to correct dimension
            voxel_data = voxel_data[slicing]

    return trimesh.voxel.VoxelGrid(voxel_data, voxel_grid.transform)

def get_shapenet_mesh(dataset, datadir='', mesh=False):
    _, model_id, obj_id = dataset.split(':')
    model_id = shapenet_category_to_id[model_id]
    obj_path = os.path.join(datadir, model_id, obj_id, 'models/model_normalized.obj')
    return as_mesh(trimesh.load(obj_path)) if mesh else trimesh.load(obj_path)
    

def get_dataset_points(dataset, datadir, n_points=1024, normalize=False):
    if dataset == 'toy_points':
        points = pd.read_csv(f'{datadir}/fullData.csv', header=None).T.to_numpy()
        
    elif dataset.split(':')[0] == "ModelNet10":
        datadir = load_modelnet()
        _, object_name, object_idx = dataset.split(':')
        obj_path = os.path.join('/'.join(datadir.split('/')[:-1]), f"ModelNet10/{object_name}/train/{object_name}_{object_idx}.off")
        print(obj_path)
        mesh = trimesh.load(obj_path)
        points = mesh.sample(n_points)
        
    elif dataset.split(':')[0] == "ShapeNet":
        _, object_name, object_idx = dataset.split(':')
        object_name = shapenet_category_to_id[object_name]
        scene_or_mesh = trimesh.load(os.path.join(datadir, f"{object_name}/{object_idx}/models/model_normalized.obj"))
        points = as_mesh(scene_or_mesh).sample(n_points)
        
    elif dataset.split(':')[0] == 'Pix3D':
        with open(os.path.join(datadir, 'pix3d.json')) as f:
            metadata = json.load(f)
        _, category, obj_id = dataset.split(':')
        img_names = [f'img/{category}/{obj_id}.png', f'img/{category}/{obj_id}.jpg']
        obj_path = None
        for obj in metadata:
            if 'img' in obj and obj['img'] in img_names:
                obj_path = obj['model']
                break
        scene_or_mesh = trimesh.load(os.path.join(datadir, obj_path))
        points = as_mesh(scene_or_mesh).sample(n_points)
        
    if normalize:
        for i in range(3):
            points[:, i] = (points[:, i] - points[:, i].min()) / (points[:, i].max() - points[:, i].min())
        
    return points

# START: Alignment and CV Operations
def getRotationMatrix(angle, axis):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)

    if axis == "x":
        return np.array([
            [1, 0, 0], 
            [0, c, -s],
            [0, s, c]
        ])
    if axis == "y":
        return np.array([
            [c, 0, s], 
            [0, 1, 0],
            [-s, 0, c]
        ])
    if axis == "z":
        return np.array([
            [c, -s, 0], 
            [s, c, 0],
            [0, 0, 1]
        ])

    
def plot_3D_paper(samples_3D, range_x=None, range_y=None, range_z=None, proj_type='perspective', colors=None, eq_range=0, cubic=True, pad=0.4, point_size=2, opacity=1):
    '''\
    Plots sets of 3D points.
    ----------
    samples_3D : List[ndarray], each of shape (n_samples, 3)
        List of Set of points.
    range_x, range_y, range_z : List[int] -> [min, max] 
        The range of the plot.
    proj_type: str
        Type of projection. ('orthographic' or 'perspective')
    Returns
    -------
    fig : go.Figure
        The plotly Figure object
    '''
    plots = []
    if not colors:
        colors = ['green', 'red', 'blue', 'black'][:len(samples_3D)]
        while len(colors) != len(samples_3D):
            colors.append('red')
    assert len(colors) == len(samples_3D), "Color must match the number of samples_3D"
    for sample_3D, color in zip(samples_3D, colors):
        plots.append(go.Scatter3d(
            x=sample_3D[:, 0],
            y=sample_3D[:, 1],
            z=sample_3D[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                opacity=opacity,
                color=sample_3D[:, 2] if color == 'fancy' else color,
                colorscale='hsv',
                symbol='circle',
            ),
            legendgroup=color,
            showlegend=True,
            name=color
        ))

    layout = go.Layout(
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
        ), 
        # autosize=True,
        scene=dict(
            xaxis_visible=False, yaxis_visible=False,zaxis_visible=False
        )
    )
    
    fig = go.Figure(data=[*plots], layout=layout)
    return fig

    
def plot_3D(samples_3D, range_x=None, range_y=None, range_z=None, proj_type='perspective', colors=None, eq_range=0, cubic=True, pad=0.4):
    '''\
    Plots sets of 3D points.
    ----------
    samples_3D : List[ndarray], each of shape (n_samples, 3)
        List of Set of points.
    range_x, range_y, range_z : List[int] -> [min, max] 
        The range of the plot.
    proj_type: str
        Type of projection. ('orthographic' or 'perspective')
    Returns
    -------
    fig : go.Figure
        The plotly Figure object
    '''
    plots = []
    if not colors:
        colors = ['green', 'red', 'blue', 'black'][:len(samples_3D)]
        while len(colors) != len(samples_3D):
            colors.append('red')
    assert len(colors) == len(samples_3D), "Color must match the number of samples_3D"
    for sample_3D, color in zip(samples_3D, colors):
        plots.append(go.Scatter3d(
            x=sample_3D[:, 0],
            y=sample_3D[:, 1],
            z=sample_3D[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                opacity=0.5,
                color=color,
                symbol='circle',
            ),
            legendgroup=color,
            showlegend=True,
            name=color
        ))

    axises = []
    
    if not range_x:
        if eq_range is None:
            range_x = [-np.inf, np.inf]
            for i in range(len(samples_3D)):
                range_x = [
                    np.min([range_x[0], samples_3D[i][:, 0].min()]),
                    np.max([range_x[1], samples_3D[i][:, 0].max()])
                ]
        else:
            range_x = [samples_3D[eq_range][:,0].min(), samples_3D[eq_range][:,0].max()]
    if not range_y:
        if eq_range is None:
            range_y = [-np.inf, np.inf]
            for i in range(len(samples_3D)):
                range_y = [
                    np.min([range_y[0], samples_3D[i][:, 1].min()]),
                    np.max([range_y[1], samples_3D[i][:, 1].max()])
                ]
        else:
            range_y = [samples_3D[eq_range][:,1].min(), samples_3D[eq_range][:,1].max()]
    if not range_z:
        if eq_range is None:
            range_z = [np.inf, -np.inf]
            for i in range(len(samples_3D)):
                range_z = [
                    np.min([range_z[0], samples_3D[i][:, 2].max()]),
                    np.max([range_z[1], samples_3D[i][:, 2].min()])
                ]
        else:
            range_z = [samples_3D[eq_range][:,2].max(), samples_3D[eq_range][:,2].min()]
            
    
    if cubic:
        range_x = np.array([np.min([range_x, range_y, range_z]), np.max([range_x, range_y, range_z])])
        range_x += np.abs(range_x)*[-pad, pad]
        range_y = range_x.copy()
        range_z = range_x.copy()[::-1]
        
    axises.append(go.Scatter3d(x=range_x, y=[0, 0], z=[0, 0], mode='lines', showlegend=True, name="Axis", legendgroup="axis"))
    axises.append(go.Scatter3d(x=[0, 0], y=range_y, z=[0, 0], mode='lines', showlegend=False, name="Axis", legendgroup="axis"))
    axises.append(go.Scatter3d(x=[0, 0], y=[0, 0], z=range_z, mode='lines', showlegend=False, name="Axis", legendgroup="axis"))

    layout = go.Layout(
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
        ), autosize=True,
        scene=dict(
            camera=dict(
                up=dict(x=0, y=1, z=0), 
                center=dict(x=0, y=0, z=0),
                eye=dict({'x': 0, 'y': 0, 'z': 1}),
                projection=dict(type=proj_type)
            ),
            xaxis = dict(nticks=4, range=range_x),
            yaxis = dict(nticks=4, range=range_y),
            zaxis = dict(nticks=4, range=range_z),
            aspectmode='cube'
        )
    )
    
    fig = go.Figure(data=[*plots, *axises], layout=layout)
    return fig


def apply_rotation(points, rot_mat):
    return np.dot(points, rot_mat)

def create_rotated_points(points):
    for angle in [45, 90, 90+45, 180, -45, -90, -90-45, -180]:
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        # for rot around x
        rot_mat1 = np.array([
            [1, 0, 0], 
            [0, c, -s],
            [0, s, c]
        ])
        # for rot around y
        rot_mat2 = np.array([
            [c, 0, s], 
            [0, 1, 0], 
            [-s, 0, c]
        ])
        # for rot around z
        rot_mat3 = np.array([
            [c, -s, 0], 
            [s, c, 0], 
            [0, 0, 1]
        ])
        yield from [apply_rotation(points, rot_mat) for rot_mat in [rot_mat1, rot_mat2, rot_mat3]]
        
def plotly_fig2array(fig):
    #convert Plotly fig to an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def apply_transformation(points, trans_mat):
    initial_shape = points.shape
    if trans_mat.shape[0] == 3:
        trans_mat = t_to_homo(trans_mat)
    if initial_shape[1] == 3:
        points = points_to_homo(points)
    ret = np.dot(trans_mat, points.T).T
    if initial_shape[1] == 3:
        ret = points_to_non_homo(ret)
    return ret

# Calculate reconstruction metrics
def t_to_homo(mat):
    if mat.shape[0] == 3:
        mat = np.append(mat, [[0, 0, 0]], axis=0)
        mat = np.insert(mat, 3, [0, 0, 0, 1], axis=1)
    return mat

def points_to_homo(points):
    if points.shape[1] == 4:
        return points
    return np.append(points, np.ones((len(points), 1)), axis=1)

def points_to_non_homo(points):
    if points.shape[1] == 3:
        return points
    return points[:, :3]