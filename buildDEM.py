import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
# 读取数据
points = []
with open('points.txt', 'r') as file:
    next(file)  # 跳过第一行
    for line in file:
        points.append([float(value) for value in line.split()])

# 将数据分成 x, y, z 三个部分
points = np.array(points)
x, y, z = points[:, 0], points[:, 1], points[:, 2]

# 定义网格分辨率
grid_resolution = 15  # 15米分辨率

# 创建网格
grid_x, grid_y = np.mgrid[min(x):max(x):grid_resolution, min(y):max(y):grid_resolution]


# 手动实现移动曲面拟合法进行插值
def moving_surface_fitting(x, y, z, grid_x, grid_y, initial_radius=30, min_neighbors=6):
    grid_z = np.zeros(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            radius = initial_radius
            while True:
                # 找到当前网格点的邻近点
                distances = np.sqrt((x - grid_x[i, j]) ** 2 + (y - grid_y[i, j]) ** 2)
                neighbors = distances < radius
                if np.sum(neighbors) > min_neighbors:
                    break
                radius *= 1.5  # 增大半径
            if np.sum(neighbors) > 0:
                # 使用邻近点拟合一个局部平面
                A = np.c_[x[neighbors], y[neighbors], np.ones(np.sum(neighbors))]
                W = np.diag(1 / distances[neighbors]**2)  # 计算权重矩阵

                # 手动实现加权最小二乘法
                W_A = np.dot(W, A)
                W_z = np.dot(W, z[neighbors])
                A_T_W_A = np.dot(W_A.T, W_A)
                A_T_W_z = np.dot(W_A.T, W_z)
                C = np.linalg.solve(A_T_W_A, A_T_W_z)

                grid_z[i, j] = C[0] * grid_x[i, j] + C[1] * grid_y[i, j] + C[2]
            else:
                grid_z[i, j] = np.nan  # 如果没有邻近点，则设置为NaN
    return grid_z


# 使用移动曲面拟合法进行插值
grid_z = moving_surface_fitting(x, y, z, grid_x, grid_y)
# 保存15米分辨率的DEM为GeoTIFF
transform = from_origin(min(x), max(y), grid_resolution, grid_resolution)
with rasterio.open(
    'dem_15m_resolution.tiff',
    'w',
    driver='GTiff',
    width=grid_z.shape[0],
    height=grid_z.shape[1],
    count=1,
    dtype=grid_z.dtype,
    crs='+proj=latlong',
    transform=transform,
) as dst:
    dst.write(np.flipud(grid_z.T), 1)
# 绘制结果
plt.figure()
plt.contourf(grid_x, grid_y, grid_z, levels=256,cmap='viridis')
plt.colorbar(label='Elevation (m)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('DEM with 15m Resolution')
plt.show()

# 扩展15米分辨率的DEM
extended_grid_z = np.pad(grid_z, pad_width=1, mode='edge')

# 创建扩展后的网格
extended_grid_x, extended_grid_y = np.mgrid[
    min(x) - grid_resolution:max(x) + grid_resolution:grid_resolution,
    min(y) - grid_resolution:max(y) + grid_resolution:grid_resolution
]
# 定义新的网格分辨率
new_grid_resolution = 10  # 10米分辨率

# 创建新的网格
new_grid_x, new_grid_y = np.mgrid[min(x):max(x):new_grid_resolution, min(y):max(y):new_grid_resolution]

# 手动实现双线性插值
def bilinear_interpolation(grid_x, grid_y, grid_z, new_grid_x, new_grid_y):
    new_grid_z = np.zeros(new_grid_x.shape)
    for i in range(new_grid_x.shape[0]):
        for j in range(new_grid_x.shape[1]):
            x1 = np.searchsorted(grid_x[:, 0], new_grid_x[i, j]) - 1
            y1 = np.searchsorted(grid_y[0, :], new_grid_y[i, j]) - 1
            x2 = x1 + 1
            y2 = y1 + 1

            if x1 < 0 or y1 < 0 or x2 >= grid_x.shape[0] or y2 >= grid_y.shape[1]:
                new_grid_z[i, j] = np.nan
                continue

            Q11 = grid_z[x1, y1]
            Q21 = grid_z[x2, y1]
            Q12 = grid_z[x1, y2]
            Q22 = grid_z[x2, y2]

            if np.isnan(Q11) or np.isnan(Q21) or np.isnan(Q12) or np.isnan(Q22):
                # 对边界点进行特殊处理
                valid_points = [(Q11, x1, y1), (Q21, x2, y1), (Q12, x1, y2), (Q22, x2, y2)]
                valid_points = [p for p in valid_points if not np.isnan(p[0])]
                if len(valid_points) == 0:
                    new_grid_z[i, j] = 0  # 如果没有有效点，设置为0
                else:
                    # 使用有效点进行插值
                    weights = [1 / np.sqrt((new_grid_x[i, j] - p[1])**2 + (new_grid_y[i, j] - p[2])**2) for p in valid_points]
                    new_grid_z[i, j] = np.dot([p[0] for p in valid_points], weights) / np.sum(weights)
                continue

            x1, x2 = grid_x[x1, 0], grid_x[x2, 0]
            y1, y2 = grid_y[0, y1], grid_y[0, y2]

            new_grid_z[i, j] = (Q11 * (x2 - new_grid_x[i, j]) * (y2 - new_grid_y[i, j]) +
                                Q21 * (new_grid_x[i, j] - x1) * (y2 - new_grid_y[i, j]) +
                                Q12 * (x2 - new_grid_x[i, j]) * (new_grid_y[i, j] - y1) +
                                Q22 * (new_grid_x[i, j] - x1) * (new_grid_y[i, j] - y1)) / ((x2 - x1) * (y2 - y1))
    return new_grid_z

# 使用双线性插值法将15米分辨率的DEM变成10米分辨率的DEM
new_grid_z = bilinear_interpolation(extended_grid_x, extended_grid_y, extended_grid_z, new_grid_x, new_grid_y)# 绘制结果
plt.figure()
plt.contourf(new_grid_x, new_grid_y, new_grid_z, levels=255, cmap='viridis')
plt.colorbar(label='Elevation (m)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('DEM with 10m Resolution')
plt.show()
# 保存10米分辨率的DEM为GeoTIFF
transform = from_origin(min(x), max(y), new_grid_resolution, new_grid_resolution)
with rasterio.open(
    'dem_10m_resolution.tiff',
    'w',
    driver='GTiff',
    width=new_grid_z.shape[0],
    height=new_grid_z.shape[1],
    count=1,
    dtype=new_grid_z.dtype,
    crs='+proj=latlong',
    transform=transform,
) as dst:
    dst.write(np.flipud(new_grid_z.T), 1)