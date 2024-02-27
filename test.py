import coloredlogs
import hydra
import logging
import torch.backends.cudnn as cudnn
import torch.cuda
from pytorch3d.loss import chamfer_distance
from thop import profile, clever_format
from tqdm import tqdm
import numpy as np
import time

from models import MSNet2D, MSNet3D, calc_IoU, eval_metric
from datasets import VoxelDSDatasetCalib

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

BATCH_SIZE = 1
MAXDISP = 192

cudnn.benchmark = True

test_dataset = VoxelDSDatasetCalib('/work/vig/Datasets/DrivingStereo',
                                   './filenames/DS_test_gt_calib.txt',
                                   False,
                                   [-8, 10, -3, 3, 0, 30],
                                   [3, 1.5, 0.75, 0.375])
# TestImgLoader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False, prefetch_factor=4)
model = MSNet2D(MAXDISP)


# model = MSNet3D(MAXDISP)


def calc_voxel_grid(filtered_cloud, grid_size, voxel_size):
    # quantized point values, here you will loose precision
    xyz_q = np.floor(np.array(filtered_cloud / voxel_size)).astype(int)
    # Empty voxel grid
    vox_grid = np.zeros(grid_size)
    offsets = np.array([8 / voxel_size, 3 / voxel_size, 0])
    xyz_offset_q = np.clip(xyz_q + offsets, [0, 0, 0], np.array(grid_size) - 1)
    # Setting all voxels containitn a points equal to 1
    vox_grid[xyz_offset_q[:, 0], xyz_offset_q[:, 1], xyz_offset_q[:, 2]] = 1

    # get back indexes of populated voxels
    xyz_v = np.asarray(np.where(vox_grid == 1))
    cloud_np = np.asarray([(pt - offsets) * voxel_size for pt in xyz_v.T])
    return torch.from_numpy(vox_grid), cloud_np


def eval_model():
    if torch.cuda.is_available():
        model.cuda()

    state_dict = torch.load('./checkpoints/MSNet2D_DS.ckpt')['model']
    # state_dict = torch.load('./checkpoints/MSNet3D_DS.ckpt')['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k[7:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    iou_dict = MetricDict()
    cd_dict = MetricDict()
    infer_time = []
    for batch_idx, sample in enumerate(tqdm(test_dataset)):
        imgL = sample['left'][None, ...]
        imgR = sample['right'][None, ...]
        voxel_gt = sample['voxel_grid'][-1]

        if torch.cuda.is_available():
            imgL = imgL.cuda()
            imgR = imgR.cuda()

        start = time.time()
        with torch.no_grad():
            disp_est = model(imgL, imgR)[-1].squeeze().cpu().numpy()
            assert len(disp_est.shape) == 2
            disp_est[disp_est <= 0] -= 1.

        depth_est = test_dataset.f_u * 0.54 / disp_est
        cloud_est = test_dataset.calc_cloud(depth_est)
        filtered_cloud_est = test_dataset.filter_cloud(cloud_est)
        voxel_est, _ = calc_voxel_grid(filtered_cloud_est, (48, 16, 80), .375)
        infer_time.append(time.time() - start)

        iou_dict.append(eval_metric([voxel_est], [voxel_gt], calc_IoU, depth_range=[.5, 1.]))
        cd_dict.append(eval_metric([voxel_est], [voxel_gt], eval_cd, [0.375], depth_range=[.5, 1.]))

    iou_mean = iou_dict.mean()
    cd_mean = cd_dict.mean()

    for k in iou_mean.keys():
        msg = f'Depth - {k}: IoU = {str(iou_mean[k].tolist())}; CD = {str(cd_mean[k].tolist())}'
        logger.info(msg)
    avg_infer = np.mean(np.array(infer_time))
    logger.info(f'Avg_infer = {avg_infer}; FPS = {1 / avg_infer}')


def eval_cd(pred, gt, scale):
    pred_coord = torch.nonzero((pred.squeeze(0) >= 0.5).int()) * float(scale)
    gt_coord = torch.nonzero((gt.squeeze(0) == 1).int()) * float(scale)

    return chamfer_distance(pred_coord[None, ...], gt_coord[None, ...])[0]


class MetricDict:
    def __init__(self):
        self._data = {}

    def append(self, in_dict):
        for k, v, in in_dict.items():
            if k not in self._data:
                self._data[k] = [v]
            else:
                self._data[k].append(v)

    def mean(self):
        out_dict = {}
        for k, v in self._data.items():
            v_t = torch.asarray(v)
            out_dict[k] = torch.mean(v_t, dim=0)

        return out_dict

    def __getattr__(self, item):
        return getattr(self._data, item)()

    def __getitem__(self, item):
        return self._data[item]


def eval_ops():
    if torch.cuda.is_available():
        model.cuda()

    sample = test_dataset[0]
    imgL, imgR, voxel_gt = sample['left'][None, ...], sample['right'][None, ...], sample['voxel_grid']
    if torch.cuda.is_available():
        imgL = imgL.cuda()
        imgR = imgR.cuda()
    with torch.no_grad():
        macs, params = clever_format(profile(model, inputs=(imgL, imgR)), '%.3f')

    print(f'MACS: {macs}, PARAMS: {params}')


if __name__ == '__main__':
    eval_model()
    # eval_ops()
