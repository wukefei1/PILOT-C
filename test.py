
import numpy as np
import struct
from utils import *
    
import numpy as np
from scipy.fft import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# 输入信号
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
t1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
x = np.array([0, 2, 4, 6, 8, 10, 10, 10, 10, 10, 8, 6, 4, 2, 0]) * 2
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 10
sumx = np.cumsum(x)
sumx = np.append(np.float64(0), sumx)
sumy = np.cumsum(y)
sumy = np.append(np.float64(0), sumy)

mpl.rcParams['font.size'] = 24


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(sumx, sumy, t, linewidth=3)
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])
# ax.set_xlabel('X', labelpad=-15)
# ax.set_ylabel('Y', labelpad=-15)
# ax.set_zlabel('T', labelpad=-15)
# plt.savefig("images/trajectory.png", dpi=300, bbox_inches='tight')

# fig, ax = plt.subplots(figsize=(8, 4))

labels = ['', '', r'$\leftarrow$ low-frequency', '', '', '', '', '', '', '', '', '', r'high-frequency $\rightarrow$', '', '']

# plt.plot(t1, x, linewidth=3)
# plt.xticks([0, 5, 10, 15])
# plt.yticks([0, 10, 20])
# plt.xlabel("T (s)")
# plt.ylabel("V (m/s)")
# plt.subplots_adjust(left=0.12, right=0.88, top=0.95, bottom=0.2)
# plt.savefig("images/velocity_time_curve.pdf", dpi=300)


# plt.plot(t1, dctn(x) / 100, linewidth=3)
# plt.scatter(t1[0:3], dctn(x)[0:3] / 100, marker='x', color='r', linewidths=3, s=200)
# plt.xticks(t1, labels)
# # plt.xlabel("T")
# plt.ylabel(r"DCT V ($\times$ 100)")
# plt.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.2)
# plt.savefig("images/DCT_result.pdf", dpi=300)





error = 2
scale = 0.3 / error

x = x - x.mean()
y = y - y.mean()

# N = len(x)
dctn_x = dctn(x)
dctn_y = dctn(y)
print(x.tolist(), y.tolist())
print(dctn_x.tolist(), dctn_y.tolist())
plt.xticks([])
plt.yticks([])


# plt.plot(sumx, sumy)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("images/trajectory_xy.png", dpi=300)


# plt.plot(t, sumx, linewidth=3)
# plt.savefig("images/trajectory_xt.png", dpi=300, bbox_inches='tight')

# plt.plot(t, sumy, linewidth=3)
# plt.savefig("images/trajectory_yt.png", dpi=300, bbox_inches='tight')

# plt.plot(t1, x, linewidth=3)
# plt.savefig("images/VX.png", dpi=300, bbox_inches='tight')

# plt.plot(t1, y, linewidth=3)
# plt.savefig("images/VY.png", dpi=300, bbox_inches='tight')

# plt.hlines(y=0, xmin=0, xmax=15)


# plt.plot(t1, dctn_x, linewidth=3)
# plt.savefig("images/dctX.png", dpi=300, bbox_inches='tight')

# plt.plot(t1, dctn_y, linewidth=3)
# plt.savefig("images/dctY.png", dpi=300, bbox_inches='tight')




print(dctn_x, dctn_y)

round_dctn_x = np.round(dctn_x * scale)
round_dctn_y = np.round(dctn_y * scale)

# for i in range(4, len(round_dctn_x)):
#     round_dctn_x[i] = 0
#     round_dctn_y[i] = 0

print(round_dctn_x.tolist(), round_dctn_y.tolist())

# plt.plot(t1, dctn_x, linewidth=3)
# plt.plot(t1, round_dctn_x / scale, linewidth=3)
# plt.savefig("images/round_dctX.png", dpi=300, bbox_inches='tight')

# plt.plot(t1, dctn_y, linewidth=3)
# plt.plot(t1, round_dctn_y / scale, linewidth=3)
# plt.savefig("images/round_dctY.png", dpi=300, bbox_inches='tight')

idctx = idctn(round_dctn_x / scale)
idcty = idctn(round_dctn_y / scale)

# plt.plot(t1, idctx, linewidth=3)
# plt.savefig("images/IDCT_VX.png", dpi=300, bbox_inches='tight')

# plt.plot(t1, idcty, linewidth=3)
# plt.savefig("images/IDCT_VY.png", dpi=300, bbox_inches='tight')

sum_idctx = np.cumsum(idctx)
sum_idctx = np.append(np.float64(0), sum_idctx)
sum_idcty = np.cumsum(idcty)
sum_idcty = np.append(np.float64(0), sum_idcty)

# plt.plot(t, sum_idctx, linewidth=3)
# plt.savefig("images/decompressed_x.png", dpi=300, bbox_inches='tight')

# plt.plot(t, sum_idcty, linewidth=3)
# plt.savefig("images/decompressed_y.png", dpi=300, bbox_inches='tight')

error_index = np.where(np.abs(sum_idctx - sumx) > error)[0]

# plt.plot(sum_idctx, sum_idcty, label="reconstructed", linewidth=2)
# plt.plot(sumx, sumy, label="origin", linewidth=2)
# plt.scatter(sumx[error_index], sumy[error_index], label="error", marker="x", color="red")
# plt.legend()
# plt.savefig("images/trajectory_reconstructed.png", dpi=300, bbox_inches='tight')

# sum_idctx[1] -= 4
# sum_idctx[14] += 4
# sum_idctx[2] -= 4
# sum_idctx[13] += 4
# sum_idctx[5] += 4
# sum_idctx[10] -= 4
# sum_idctx[6] += 4
# sum_idctx[9] -= 4


print(sum_idctx, sum_idcty)
print(sum_idctx - sumx, sum_idcty - sumy)


# plt.plot(sum_idctx, sum_idcty, label="reconstructed", linewidth=2)
# plt.plot(sumx, sumy, label="origin", linewidth=2)
# plt.legend()
# plt.savefig("images/trajectory_removed_error.png", dpi=300, bbox_inches='tight')

for i in range(-10, 10):
    assert bitarray2zigzag(zigzag2bitarray(i), 0)[0] == i


# def compress():
#     traj = np.concatenate((sumx.reshape(-1, 1), sumy.reshape(-1, 1)), axis=1)
#     t_list = np.arange(len(traj))
    
#     traj_dim = traj.shape[1]
#     max_error = error
#     accuracy = 0.2
#     max_accuracy_error = accuracy * max_error
#     polygon_num = 16
    
#     max_accuracy_error = bitarray2float(float2bitarray(max_accuracy_error))
#     max_error_dim = max_accuracy_error / math.sqrt(2)
    
#     max_error = max_error - max_accuracy_error
    
#     from shapely.geometry import Polygon, Point
    
#     point_list = [traj[0]]
#     index_list = [0]
#     polygon = Polygon()
#     for i in range(1, len(traj)):
#         def get_polygon(point0, point1, index0, index1, r):
#             c = (t_list[index0 + 1] - t_list[index0]) / (t_list[index1] - t_list[index0])
#             point = point0 + (point1 - point0) * c
#             point_list = []
#             for j in range(polygon_num):
#                 angle = (2 * j - 1) * np.pi / polygon_num
#                 point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) * c)
#             return Polygon(point_list)
        
#         temp_polygon = get_polygon(point_list[-1], traj[i], index_list[-1], i, max_error)
#         if polygon.is_empty:
#             polygon = temp_polygon
#         else:
#             center_point = np.array([polygon.centroid.x, polygon.centroid.y])
#             new_polygon = polygon.intersection(temp_polygon)
#             if new_polygon.is_empty:
#                 c = (t_list[index_list[-1] + 1] - t_list[index_list[-1]]) / (t_list[i - 1] - t_list[index_list[-1]])
#                 add_point = traj[i - 1]
#                 temp_point = point_list[-1] + (add_point - point_list[-1]) * c
#                 if not polygon.contains(Point(temp_point)):
#                     add_point = point_list[-1] + (center_point - point_list[-1]) / c
                
#                 point_list.append(add_point)
#                 index_list.append(i - 1)
#                 polygon = get_polygon(add_point, traj[i], index_list[-1], i, max_error)
#             else:
#                 polygon = new_polygon
    
#     add_point = traj[-1]
#     if not polygon.is_empty:
#         center_point = np.array([polygon.centroid.x, polygon.centroid.y])
#         c = (t_list[index_list[-1] + 1] - t_list[index_list[-1]]) / (t_list[len(traj) - 1] - t_list[index_list[-1]])
#         temp_point = point_list[-1] + (add_point - point_list[-1]) * c
#         if not polygon.contains(Point(temp_point)):
#             add_point = point_list[-1] + (center_point - point_list[-1]) / c
    
#     point_list.append(add_point)
#     index_list.append(len(traj) - 1)
    
#     return point_list, index_list
    
# point_list, index_list = compress()
# print(point_list, index_list)

# import matplotlib.pyplot as plt
 
# # 示例数据
# x = range(1, 6)
# y1 = [2, 3, 5, 7, 11]  # 折线图数据
# y2 = [1, 2, 4, 6, 10]  # 柱状图数据
 
# # 创建图表和轴
# fig, ax1 = plt.subplots()
 
# # 绘制柱状图
# color = 'tab:blue'
# ax1.set_xlabel('x label')
# ax1.set_ylabel('y1 label', color=color)
# ax1.bar(x, y2, color=color, alpha=0.6, label='Bar')
# ax1.tick_params(axis='y', labelcolor=color)
 
# # 创建第二个y轴
# ax2 = ax1.twinx()  
# color = 'tab:red'
# ax2.set_ylabel('y2 label', color=color)
# ax2.plot(x, y1, color=color, marker='o', label='Line')
# ax2.tick_params(axis='y', labelcolor=color)
 
# # 添加图例
# fig.tight_layout()
# plt.legend(loc='upper left')
# plt.savefig('test.pdf')

import matplotlib.pyplot as plt
 
# 示例数据
x = range(1, 6)
y1 = [20, 35, 30, 35, 27]
y2 = [1, 2, 0.7, 2.5, 3]
fig, ax1 = plt.subplots()

ax1.plot(x, y2, 'r', label='plot')
ax1.set_ylabel('plot data')
ax1.tick_params(axis='y')
ax1.set_xlabel('x')

ax2 = ax1.twinx()
ax2.bar(x, y1, color='b', alpha=0.6, label='bar')
ax2.set_ylabel('bar data')
ax2.tick_params(axis='y')
 
# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines.extend(lines2)
labels.extend(labels2)
fig.legend(lines, labels, loc = 'upper center', ncol=4, bbox_to_anchor=(0.5, 1))
 
# 设置标题和标签
plt.title('test')
 
plt.savefig('test.pdf')