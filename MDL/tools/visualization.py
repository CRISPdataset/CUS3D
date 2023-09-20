import argparse
import os
import os.path as osp
from operator import itemgetter

import numpy as np

# yapf:disable
COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3) * 255
# yapf:enable

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array([
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
    'picture', 'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink',
    'bathtub','otherfurniture'
    
])
#otherfurniture

CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    # 'picture': [0, 160, 55],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160],
    'whiteboard': [210,105,30], #深绿色,
    'printer': [79,79,79], #灰色,
    'dustbin':[	118, 238, 198],#薄荷绿
    'computer': [205, 92, 92],#印第安红,
    'ball':[210,105,30], #巧克力橙
    'stereo':[205,205,193	],#浅灰,
    'running machine': [139,34,82]
    
}
SEMANTIC_IDX2NAME = {
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    14: 'desk',
    16: 'curtain',
    24: 'refridgerator',
    28: 'shower curtain',
    33: 'toilet',
    34: 'sink',
    36: 'bathtub',
    39: 'otherfurniture'
}


def get_coords_color(opt):
    coord_file = osp.join(opt.prediction_path, 'coords', opt.room_name + '.npy')
    color_file = osp.join(opt.prediction_path, 'colors', opt.room_name + '.npy')
    label_file = osp.join(opt.prediction_path, 'semantic_label', opt.room_name + '.npy')
    # inst_label_file = osp.join(opt.prediction_path, 'gt_instance', opt.room_name + '.txt')
    xyz = np.load(coord_file)
    rgb = np.load(color_file)
    label = np.load(label_file)
    # inst_label = np.array(open(inst_label_file).read().splitlines(), dtype=int)
    # inst_label = inst_label % 1000 - 1
    rgb = (rgb + 1) * 127.5

    if (opt.task == 'semantic_gt'):
        label = label.astype(int)
        label_rgb = np.zeros(rgb.shape)
        # print(np.array(
        #     itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR)))
        label_rgb[label >= 0, :3] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

    elif (opt.task == 'semantic_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb
    elif (opt.task == 'clip_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'clip_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        rgb = label_pred_rgb

    sem_valid = (label != -100)
    xyz = xyz[sem_valid]
    rgb = rgb[sem_valid]

    return xyz, rgb

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write('{:f} {:f} {:f} {:d} {:d} {:d}\n'.format(vert[0], vert[1], vert[2],
                                                            int(color[0] * 255),
                                                            int(color[1] * 255),
                                                            int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prediction_path', help='path to the prediction results', default='./results')
    parser.add_argument('--room_name', help='room_name', default='scene0011_00')
    parser.add_argument(
        '--task',
        help='input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred/clip_pred',
        default='instance_pred')
    parser.add_argument('--out', help='output point cloud file in FILE.ply format')
    parser.add_argument('--all', default='')
    parser.add_argument('--save_name', default='debug')
    opt = parser.parse_args()
    names = [opt.room_name]
    
    if opt.all:
        # 可视化所有的
        names = os.listdir(opt.all)
        names = [name.replace('.npy', '') for name in names]
    from tqdm import tqdm
    os.makedirs(f"vis/{opt.save_name}", exist_ok=True)
    for room_name in tqdm(names):
        opt.room_name = room_name
        opt.out = f"vis/{opt.save_name}/{room_name}_{opt.task}.ply"
        xyz, rgb = get_coords_color(opt)
        points = xyz[:, :3]
        colors = rgb / 255

        if opt.out != '':
            assert '.ply' in opt.out, 'output cloud file should be in FILE.ply format'
            write_ply(points, colors, None, opt.out)
        else:
            import open3d as o3d
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points)
            pc.colors = o3d.utility.Vector3dVector(colors)

            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pc)
            vis.get_render_option().point_size = 1.5
            vis.run()
            vis.destroy_window()
