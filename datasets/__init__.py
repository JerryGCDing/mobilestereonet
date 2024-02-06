from .dataset import SceneFlowDataset, KITTIDataset, DrivingStereoDataset, VoxelDSDatasetCalib

__datasets__ = {
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
}
