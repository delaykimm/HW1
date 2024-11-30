# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# 1. 평면으로 투영하는 함수
def transform_to_plane_based_coordinates(points, a, b, c, d):
    """
    평면을 새로운 XY 평면으로 간주하고,
    모든 포인트를 새로운 좌표계로 변환합니다.
    새로운 X, Y는 평면 위에서의 투영된 좌표이며,
    Z는 평면까지의 수직 거리입니다.

    Args:
        points (numpy.ndarray): 입력 포인트 클라우드, 크기 (N, 3).
        a, b, c, d (float): 평면 방정식의 계수 (ax + by + cz + d = 0).

    Returns:
        numpy.ndarray: 새로운 좌표계의 포인트 클라우드, 크기 (N, 3).
    """
    print("[DEBUG] Transforming points to plane-based coordinate system...")
    print(f"[DEBUG] Input points shape: {points.shape}")

    # 평면의 법선 벡터 및 크기 계산
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)

    # 점들을 평면에 투영
    distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_norm**2
    projected_points = points - distances[:, np.newaxis] * normal  # 투영된 좌표 계산

    # 새로운 Z 값: 평면까지의 수직 거리 (부호 유지)
    plane_equation_values = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    new_z = plane_equation_values / normal_norm

    # 새로운 좌표계
    new_x = projected_points[:, 0]
    new_y = projected_points[:, 1]
    transformed_points = np.stack((new_x, new_y, new_z), axis=-1)

    print(f"[DEBUG] Transformed points shape: {transformed_points.shape}")
    return transformed_points

# 3. 높이 맵 계산 함수
def calculate_height_map(points, x_bins, y_bins, threshold=0.2):
    print("[DEBUG] Calculating height map...")

    # Step 1: Digitize indices for x and y
    x_indices = np.digitize(points[:, 0], bins=x_bins) - 1
    y_indices = np.digitize(points[:, 1], bins=y_bins) - 1

    # Step 2: Create a mask for valid indices
    valid_mask = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                 (y_indices >= 0) & (y_indices < len(y_bins) - 1)
    valid_points = points[valid_mask]
    x_indices, y_indices = x_indices[valid_mask], y_indices[valid_mask]

    print(f"[DEBUG] Valid points count: {len(valid_points)}")

    # Step 3: Calculate initial height map
    height_map = {}
    for (z, i, j) in zip(valid_points[:, 2], x_indices, y_indices):
        key = (i, j)
        if key in height_map:
            height_map[key].append(z)
        else:
            height_map[key] = [z]

    for key, z_values in height_map.items():
        if z_values:
            sorted_z_values = np.sort(z_values)
            index_5th_percentile = max(0, int(len(sorted_z_values) * 0.05))
            height_map[key] = sorted_z_values[index_5th_percentile]
        else:
            height_map[key] = None

    print(f"[DEBUG] Initial height map size: {len(height_map)}")

    # Step 4: Post-processing based on neighbor differences
    # Step 4.1: Build KDTree only for non-empty grid points
    # Build grid keys and values
    grid_keys = [key for key in height_map.keys() if height_map[key] is not None]  # 리스트로 유지
    grid_values = np.array([height_map[key] for key in grid_keys])  # 값은 numpy 배열로 생성

    # Precompute grid centers
    grid_centers = np.array([((x_bins[i] + x_bins[i + 1]) / 2, (y_bins[j] + y_bins[j + 1]) / 2) for i, j in grid_keys])
    kdtree = cKDTree(grid_centers)

    processed_height_map = {}
    search_radius = 1  # Define search radius for neighbor checks

    for (i, j), height in height_map.items():
        if height is None:
            # Skip empty grids
            continue

        # Get the center of the current grid
        center_x = (x_bins[i] + x_bins[i + 1]) / 2
        center_y = (y_bins[j] + y_bins[j + 1]) / 2

        # Find neighbors within the search radius
        indices = kdtree.query_ball_point([center_x, center_y], r=search_radius)
        neighbor_heights = [grid_values[idx] for idx in indices if grid_values[idx] is not None]

        # Process the height difference
        if neighbor_heights:
            smallest_neighbor = min(neighbor_heights)
            if abs(height - smallest_neighbor) > threshold:
                processed_height_map[(i, j)] = smallest_neighbor
            else:
                processed_height_map[(i, j)] = height
        else:
            processed_height_map[(i, j)] = height  # Leave unchanged if no neighbors found

    print(f"[DEBUG] Processed height map size: {len(processed_height_map)}")
    return processed_height_map

def cluster_non_floor_points(non_floor_pcd, eps=0.5, min_points=10):
    """
    DBSCAN 클러스터링을 사용하여 비바닥(non-floor) 점을 군집화합니다.
    
    Args:
        non_floor_pcd (open3d.geometry.PointCloud): 비바닥 포인트 클라우드.
        eps (float): DBSCAN의 클러스터 반경.
        min_points (int): 클러스터를 형성하기 위한 최소 포인트 수.
        
    Returns:
        list of open3d.geometry.PointCloud: 각 클러스터에 해당하는 포인트 클라우드 리스트.
    """
    print("[DEBUG] Performing DBSCAN clustering...")
    labels = np.array(non_floor_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    print(f"[DEBUG] Number of clusters: {max_label + 1}")
    
    clusters = []
    for cluster_id in range(max_label + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_pcd = non_floor_pcd.select_by_index(cluster_indices)
        clusters.append(cluster_pcd)
    
    return clusters, labels

# 4. 시각화 함수
def visualize_point_clouds(pcd_list, window_name="ROR Visualization", point_size=0.5):
    # 단일 객체를 리스트로 변환
    if isinstance(pcd_list, o3d.geometry.PointCloud):
        pcd_list = [pcd_list]
        
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 메인 실행 코드
# file_path = "data/05_straight_duck_walk/pcd/pcd_000370.pcd"
# file_path = "data/01_straight_walk/pcd/pcd_000250.pcd"
file_path = "data/04_zigzag_walk/pcd/pcd_000267.pcd"
# file_path = "data/06_straight_crawl/pcd/pcd_000500.pcd"
# file_path = "data/02_straight_duck_walk/pcd/pcd_000500.pcd"
# file_path = "data/03_straight_crawl/pcd/pcd_000900.pcd"
# file_path = "data/07_straight_walk/pcd/pcd_000350.pcd"
original_pcd = o3d.io.read_point_cloud(file_path)
print(f"[DEBUG] Original point cloud size: {len(original_pcd.points)}")

# Voxel Downsampling
voxel_size = 0.05
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"[DEBUG] Downsampled point cloud size: {len(downsample_pcd.points)}")

# Radius Outlier Removal (ROR)
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
# ROR 후 남은 점 (inliers)
ror_inliers_pcd = downsample_pcd.select_by_index(ind)
ror_inliers_pcd.paint_uniform_color([0, 1, 0])  # 녹색 (남은 점)

# ROR로 제거된 점 (outliers)
ror_outliers_pcd = downsample_pcd.select_by_index(ind, invert=True)
ror_outliers_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (제거된 점)

ror_pcd = downsample_pcd.select_by_index(ind)
print(f"[DEBUG] Point cloud size after ROR: {len(ror_pcd.points)}")

# 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=2000)
[a, b, c, d] = plane_model
print(f"[DEBUG] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

ror_points = np.asarray(ror_pcd.points)
transformed_points = transform_to_plane_based_coordinates(ror_points, a, b, c, d)

# 격자 생성
grid_resolution = 0.3
x_min, y_min = np.min(transformed_points[:, :2], axis=0)
x_max, y_max = np.max(transformed_points[:, :2], axis=0)

x_bins = np.arange(x_min, x_max, grid_resolution)
y_bins = np.arange(y_min, y_max, grid_resolution)
print(f"[DEBUG] Number of x_bins: {len(x_bins)}, Number of y_bins: {len(y_bins)}")

# 높이 맵 계산 및 보완
height_map = calculate_height_map(transformed_points, x_bins, y_bins, threshold = 0.05) # threshold : 주변 격자와의 높이차 

# 비바닥 점 필터링
threshold = 0.15 # threshold : 격자의 바닥 높이와의 차
non_floor_indices = []
# for idx, (x, y, z) in enumerate(projected_points):
for idx, (x, y, z) in enumerate(transformed_points):
    x_idx = np.searchsorted(x_bins, x, side='right') - 1
    y_idx = np.searchsorted(y_bins, y, side='right') - 1
    if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
        smoothed_height = height_map.get((x_idx, y_idx), None)
        if smoothed_height is not None and abs(z - smoothed_height) > threshold:
            non_floor_indices.append(idx)
print(f"[DEBUG] Total non-floor points: {len(non_floor_indices)}")

# Floor points count
# floor_indices = set(range(len(projected_points))) - set(non_floor_indices)
floor_indices = set(range(len(transformed_points))) - set(non_floor_indices)
print(f"[DEBUG] Total floor points: {len(floor_indices)}")

# 비바닥 및 바닥 포인트
non_floor_pcd = ror_inliers_pcd.select_by_index(non_floor_indices)
floor_pcd = ror_inliers_pcd.select_by_index(non_floor_indices, invert=True)

# 비바닥 점 (non-floor) 개수
num_non_floor_points = len(non_floor_pcd.points)
print(f"Number of non-floor points: {num_non_floor_points}")

# 바닥 점 (floor) 개수
num_floor_points = len(floor_pcd.points)
print(f"Number of floor points: {num_floor_points}")

# 색상 설정
floor_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
non_floor_pcd.paint_uniform_color([0, 1, 0])  # 녹색

# visualize_point_clouds([floor_pcd, non_floor_pcd], 
#                        window_name="Floor (Red) and Non-Floor (Green) Points", point_size=1.0)

# 클러스터링 실행
eps = 0.3  # 클러스터 반경
min_points = 30  # 클러스터 최소 크기
clusters, cluster_labels = cluster_non_floor_points(non_floor_pcd, eps=eps, min_points=min_points)

# 클러스터링 결과 시각화 (각 클러스터 다른 색상 적용)
for i, cluster in enumerate(clusters):
    color = np.random.rand(3)  # 무작위 색상
    cluster.paint_uniform_color(color)

visualize_point_clouds(clusters, window_name="Clustered Non-Floor Points", point_size=1.0)