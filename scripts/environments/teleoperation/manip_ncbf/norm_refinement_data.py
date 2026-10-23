import os
import numpy as np
from models import NCBF, Normalizer
from helpers import normalize_data

# 指定relabel后的npz文件路径和输出目录
relabeled_data_path = "boundary_check_results/final_updated_dataset.npz"
output_dir = "boundary_check_results"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
print(f"Loading relabeled data from {relabeled_data_path}...")
data = np.load(relabeled_data_path, allow_pickle=True)

# 提取特征数据
safe_local_xs = data['safe_local_xs']
safe_local_nxs = data['safe_local_nxs']
safe_ee_xs = data['safe_ee_xs'] 
safe_ee_nxs = data['safe_ee_nxs']
unsafe_local_xs = data['unsafe_local_xs']
unsafe_ee_xs = data['unsafe_ee_xs']

print(f"Loaded relabeled data:")
print(f"  safe_local_xs shape: {safe_local_xs.shape}")
print(f"  safe_local_nxs shape: {safe_local_nxs.shape}")
print(f"  safe_ee_xs shape: {safe_ee_xs.shape}")
print(f"  safe_ee_nxs shape: {safe_ee_nxs.shape}")
print(f"  unsafe_local_xs shape: {unsafe_local_xs.shape}")
print(f"  unsafe_ee_xs shape: {unsafe_ee_xs.shape}")

# 提取特征
print("\nExtracting features...")
safe_h_obj_features = np.zeros((safe_local_xs.shape[0], safe_local_xs.shape[1], 4), dtype=np.float32)
for i in range(safe_local_xs.shape[0]):
    for j in range(safe_local_xs.shape[1]):
        safe_h_obj_features[i, j] = safe_local_xs[i, j, :4].astype(np.float32)

safe_h_obj_nxs_features = np.zeros((safe_local_nxs.shape[0], safe_local_nxs.shape[1], 4), dtype=np.float32)
for i in range(safe_local_nxs.shape[0]):
    for j in range(safe_local_nxs.shape[1]):
        safe_h_obj_nxs_features[i, j] = safe_local_nxs[i, j, :4].astype(np.float32)
        
unsafe_h_obj_features = np.zeros((unsafe_local_xs.shape[0], unsafe_local_xs.shape[1], 4), dtype=np.float32)
for i in range(unsafe_local_xs.shape[0]):
    for j in range(unsafe_local_xs.shape[1]):
        unsafe_h_obj_features[i, j] = unsafe_local_xs[i, j, :4].astype(np.float32)

safe_ee_xs = safe_ee_xs.astype(np.float32)
safe_ee_nxs = safe_ee_nxs.astype(np.float32)
unsafe_ee_xs = unsafe_ee_xs.astype(np.float32)

# 数据归一化
print("\nNormalizing the data...")
h_obj_inputDim = 4
ee_inputDim = 2

h_obj_normalizer = Normalizer(h_obj_inputDim)
ee_normalizer = Normalizer(ee_inputDim)

normalized_data = normalize_data(
    safe_h_obj_features, safe_h_obj_nxs_features, unsafe_h_obj_features,
    safe_ee_xs, safe_ee_nxs, unsafe_ee_xs,
    h_obj_normalizer, ee_normalizer, update=True
)

# 保存normalizer
h_obj_normalizer_path = os.path.join(output_dir, "local_normalizer")
ee_normalizer_path = os.path.join(output_dir, "arm_normalizer")

h_obj_normalizer.save_model(h_obj_normalizer_path)
ee_normalizer.save_model(ee_normalizer_path)

print(f"\nNormalizers saved at:")
print(f"  {h_obj_normalizer_path}")
print(f"  {ee_normalizer_path}")
print("Normalization completed.")