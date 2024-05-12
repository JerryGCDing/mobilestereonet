from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset, VoxelDSDatasetCalib, VoxelKITTIDataset

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
}
