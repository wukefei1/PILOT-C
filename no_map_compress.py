import numpy as np
import os
from scipy.fft import *
from utils import *
import matplotlib.pyplot as plt
import pickle
from bitarray import bitarray
import yaml
import heapq
import math
import itertools

np.random.seed(0)

if 'LINE_PROFILE_IT' not in os.environ:
    def profile(x):
        return x
    
# @profile
# def get_dp(compressed_result, length_bit, exp_bit):
#     max_bits = 0
    
#     need_bits = []
#     for i in range(len(compressed_result)):
#         if compressed_result[i] == 0:
#             need_bits.append(0)
#             continue
#         temp_bits = 1
#         while compressed_result[i] > 2 ** (temp_bits - 1) - 1 or compressed_result[i] < -2 ** (temp_bits - 1):
#             temp_bits = temp_bits + 1 
#         need_bits.append(temp_bits)
#         max_bits = max(max_bits, temp_bits)
#     max_bits = max_bits + 1
#     assert max_bits <= 2 ** exp_bit, "max_bits should not be greater than 2 ** exp_bit"
    
#     dp = np.full((len(compressed_result) + 1, max_bits), np.inf).tolist()
#     last_index = np.zeros((len(compressed_result) + 1, max_bits)).tolist()
    
#     for j in range(max_bits):
#         dp[0][j] = exp_bit + length_bit
    
#     for i in range(len(compressed_result)):
#         argmin_dp_i = np.argmin(dp[i])
#         for j in range(need_bits[i], max_bits):
#             if dp[i][argmin_dp_i] + exp_bit + length_bit + j > dp[i][j] + j:
#                 dp[i + 1][j] = dp[i][j] + j
#                 last_index[i + 1][j] = j
#             else:
#                 dp[i + 1][j] = dp[i][argmin_dp_i] + exp_bit + length_bit + j
#                 last_index[i + 1][j] = argmin_dp_i
#     for j in range(max_bits):
#         dp[0][j] = 0
    
#     return dp, last_index

# def get_compressed_result(compressed_result, dp, last_index, end_index):
#     argmin_dp = int(np.argmin(dp[end_index]))
#     temp_result = []
#     for i in range(end_index, 0, -1):
#         temp_result.append(argmin_dp)
#         argmin_dp = int(last_index[i][argmin_dp])
#     temp_result.reverse()

#     final_result = []
    
#     index = 0
#     while index < end_index:
#         temp_list = [compressed_result[index]]
#         while index + 1 < end_index and temp_result[index] == temp_result[index + 1]:
#             index = index + 1
#             temp_list.append(compressed_result[index])
#         final_result.append([temp_result[index], temp_list])
#         index = index + 1
#     return final_result

# def lossless_compress(compressed_result, length_bit, exp_bit, bits = None):
#     dp, last_index = get_dp(compressed_result, length_bit, exp_bit)
    
#     if bits is None:
#         end_index = len(compressed_result)
#     else:
#         end_index = 0
#         for i in range(len(compressed_result), 0, -1):
#             if np.min(dp[i]) <= bits:
#                 end_index = i
#                 break
    
#     final_result = get_compressed_result(compressed_result, dp, last_index, end_index)
#     return final_result, min(dp[end_index])

class Compressor():
    '''
    base class
    '''
    _cache_version = 'v20250313'
    _cache_attributes = {
        'traj',
        't_list',
    }
    
    def __init__(self, name, path, data_source):
        path_name = path.split("/")[-1]
        self.name = name
        self.data_source = data_source
        self.path = path
        self.cache_path = f'/home/wkf/data/PRESS/cache/{data_source}/{path_name}/'
        self.result_path = f'/home/wkf/data/PRESS/result/{data_source}/{path_name}/'
        self.save_path = self.result_path + f'compressed_{name}'
        self.origin_path = self.result_path + f'origin'
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if data_source == 'nuplan':
            self.load_nuplan_traj()
            self.time_accuracy = 0.01
        elif data_source == 'geolife':
            self.load_geolife_traj()
            self.time_accuracy = 1
        elif data_source == 'shangqi':
            self.load_shangqi_traj()
            self.time_accuracy = 0.01
        elif data_source == 'mopsi':
            self.load_mopsi_traj()
            self.time_accuracy = 0.001
        elif data_source == 'geolife_3d':
            self.load_geolife_3d_traj()
            self.time_accuracy = 1
        
        self.decompress_traj = None
        
    def compress(self, save=True, **kwargs):
        pass

    def decompress(self, **kwargs):
        pass
    
    def _load_cache_file(self, cache_name):
        if not os.path.exists(cache_name):
            return False
        with open(cache_name, 'rb') as f:
            cache_data = pickle.load(f)
        cache_version = cache_data['_version'] if '_version' in cache_data else 'UNKNOWN'
        if cache_version != self._cache_version:
            return False
        del cache_data['_version']
        for attr in self._cache_attributes:
            setattr(self, attr, cache_data[attr])
        return True
    
    def _save_cache_file(self, cache_name):
        cache_data = {
            '_version': self._cache_version
        }
        for attr in self._cache_attributes:
            cache_data[attr] = getattr(self, attr)
        with open(cache_name, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def load_nuplan_traj(self):
        if not self._load_cache_file(os.path.join(self.cache_path, 'cache.pkl')):
            data = [x.strip().split(",") for x in open(os.path.join(self.path, 'ego.csv')).readlines()]
            title = data[0]
            data = data[1:]
            
            def get_data(line, title, name, default = np.float64(0)):
                if name in title:
                    data = line[title.index(name)]
                    if len(data):
                        return np.float64(data)
                    else:
                        return default
                else:
                    return default
            
            traj = []
            t_list = []
            for line in data:
                x = get_data(line, title, "X_utm(m)")
                y = get_data(line, title, "Y_utm(m)")
                t = get_data(line, title, "SimTime(s)")
                t_list.append(t)
                traj.append([x, y])
            
            self.traj = np.array(traj)
            self.t_list = np.array(t_list)
            
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
            
    def load_shangqi_traj(self):
        if not self._load_cache_file(os.path.join(self.cache_path, 'cache.pkl')):
            data = [x.strip().split(",") for x in open(os.path.join(self.path, 'ego.csv')).readlines()]
            title = data[0]
            data = data[1:]
            
            def get_data(line, title, name, default = np.float64(0)):
                if name in title:
                    data = line[title.index(name)]
                    if len(data):
                        return np.float64(data)
                    else:
                        return default
                else:
                    return default
            
            traj = []
            t_list = []
            for line in data:
                x = get_data(line, title, "X_utm(m)")
                y = get_data(line, title, "Y_utm(m)")
                t = get_data(line, title, "SimTime(s)")
                t_list.append(t)
                traj.append([x, y])
            
            traj = np.array(traj)
            t_list = np.array(t_list)
            length = len(traj)
            
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
            
    def load_geolife_traj(self):
        if not self._load_cache_file(os.path.join(self.cache_path, 'cache.pkl')):
            data = [x.strip().split(",") for x in open(self.path).readlines()]
            
            traj = []
            t_list = []
            
            for line in data:
                t = np.float64(line[0])
                x = np.float64(line[1])
                y = np.float64(line[2])
                t_list.append(t)
                traj.append([x, y])
            
            self.traj = np.array(traj)
            self.t_list = np.array(t_list)
            
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
            
    def load_geolife_3d_traj(self):
        if not self._load_cache_file(os.path.join(self.cache_path, 'cache.pkl')):
            data = [x.strip().split(",") for x in open(self.path).readlines()]
            
            traj = []
            t_list = []
            
            for line in data:
                t = np.float64(line[0])
                x = np.float64(line[1])
                y = np.float64(line[2])
                z = np.float64(line[3])
                t_list.append(t)
                traj.append([x, y, z])
            
            self.traj = np.array(traj)
            self.t_list = np.array(t_list)
            
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
            
    def load_mopsi_traj(self):
        if not self._load_cache_file(os.path.join(self.cache_path, 'cache.pkl')):
            data = [x.strip().split(",") for x in open(self.path).readlines()]
            
            traj = []
            t_list = []
            
            for line in data:
                t = np.float64(line[0])
                x = np.float64(line[1])
                y = np.float64(line[2])
                t_list.append(t)
                traj.append([x, y])
            
            self.traj = np.array(traj)
            self.t_list = np.array(t_list)
            
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
            
    def evaluation_metrics(self, error_bound):
        '''
        calculate max error, mean error, compress ratio
        '''
        if error_bound == None:
            error_bound = self.error_default
        
        error = np.sqrt(np.sum((self.traj - self.decompress_traj) ** 2, axis=1))
        
        error_cnt = np.where(error > error_bound)[0]
        
        max_error = np.max(error)
        mean_error = np.mean(error)
        self.argmax_error = int(np.argmax(error))
        return {'max_error': float(max_error),
                'mean_error': float(mean_error),
                'compress_ratio': float((len(self.traj) * (len(self.traj[0]) + 1) * 8) / self.compress_bits * 8),
                'length': len(self.traj),
                'error_cnt': len(error_cnt),
                'var_error': float(np.var(error)),
                }
        
    def plot_traj(self, argmax_error=None):
        self.decompress_traj = np.array(self.decompress_traj)
        if self.traj.shape[1] == 2:
            
            if argmax_error is None:
                plt.scatter(self.traj[:, 0], self.traj[:, 1], label='origin', s=0.01)
                plt.scatter(self.decompress_traj[:, 0], self.decompress_traj[:, 1], label='decompress', s=0.01)
                path = f'images/{self.name}_total.png'
            else:
                start_index = argmax_error - 60
                end_index = argmax_error + 60
                
                plt.scatter(self.traj[start_index: end_index, 0], self.traj[start_index: end_index, 1], label='origin', s=0.1)
                plt.scatter(self.decompress_traj[start_index: end_index, 0], self.decompress_traj[start_index: end_index, 1], label='decompress', s=0.1)
                
                for i in range(start_index, end_index):  # 连接对应点
                    plt.plot([self.traj[i, 0], self.decompress_traj[i, 0]], [self.traj[i, 1], self.decompress_traj[i, 1]], 'k--', linewidth=0.1)  # 用虚线连接
                    
                path = f'images/{self.name}_local.png'
                
            plt.axis('equal')    
            plt.legend()
            plt.savefig(path, dpi=1000)
            plt.gcf().clear()
        elif self.traj.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.traj[:, 0], self.traj[:, 1], self.traj[:, 2], label='origin', s=0.1)
            ax.scatter(self.decompress_traj[:, 0], self.decompress_traj[:, 1], self.decompress_traj[:, 2], label='decompress', s=0.1)
            ax.legend()
            ax.axis('equal')
            plt.savefig(f'images/{self.name}.png', dpi=1000)
            plt.gcf().clear()
                    
        

class PILOTCCompressor(Compressor):
    def __init__(self, path, data_source):
        super().__init__('PILOT-C', path, data_source)
        self.scale_factor = 0.3
        if self.data_source == 'geolife':
            self.utf_default = 3
            self.error_default = 50
            self.a = 0.5
            self.b = 25
            self.c = 1.1
            self.time_unit = 3
        if self.data_source == 'geolife_3d':
            self.utf_default = 3
            self.error_default = 50
            self.a = 0.5
            self.b = 25
            self.c = 0.8
            self.time_unit = 3
            self.scale_factor = 0.35
        if self.data_source == 'mopsi':
            self.utf_default = 3
            self.error_default = 50
            self.a = 1
            self.b = 25
            self.c = 0.6
            self.time_unit = 2
        if self.data_source == 'nuplan':
            self.utf_default = 3
            self.error_default = 5
            self.a = 20
            self.b = 100
            self.c = 0.04
            self.time_unit = 0.1
        self.accuracy_default = 0.5
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}_block_{kwargs['block_size']}_scale_{kwargs['scale_length']}_epsilon_{kwargs['epsilon']}'
        
        max_error = bitarray2float(float2bitarray(max_error))
        block_size = int(max_error * self.a + self.b) if kwargs['block_size'] == None else kwargs['block_size']
        scale_length = min(1, self.c / math.sqrt(max_error)) if kwargs['scale_length'] == None else kwargs['scale_length']
        max_accuracy_error = accuracy * max_error
        
        self.scale_length = scale_length
        
        scale = self.scale_factor / max_error if kwargs['epsilon'] == None else 0.5 / kwargs['epsilon'] / max_error
        max_error_dim = max_error / math.sqrt(traj_dim)
        max_accuracy_error_dim = max_accuracy_error / math.sqrt(traj_dim)
        
        other_points = []
        error_points = []
        index_list = []
        
        start_index = 0
        end_index = 0
        while start_index < len(self.t_list):
            end_index = start_index
            # while end_index < len(self.t_list) - 1 and \
            #       self.t_list[end_index + 1] - self.t_list[end_index] < block_size * self.time_unit and \
            #       np.sqrt(np.sum((self.traj[end_index + 1] - self.traj[end_index]) ** 2) / (self.t_list[end_index + 1] - self.t_list[end_index])) < 200:
            #     end_index = end_index + 1
            
            
            while end_index < len(self.t_list) - 1 and \
                  self.t_list[end_index + 1] - self.t_list[end_index] < block_size * ((self.t_list[end_index] - self.t_list[start_index]) / (end_index - start_index + 1) if end_index != start_index else 1) and \
                  np.sqrt(np.sum((self.traj[end_index + 1] - self.traj[end_index]) ** 2) / (self.t_list[end_index + 1] - self.t_list[end_index])) < 200:
                end_index = end_index + 1
            
            if end_index - start_index <= 1:
                for i in range(start_index, end_index + 1):
                    other_points.append(i)
                start_index = end_index + 1
                continue
            index_list.append([start_index, end_index])
            start_index = end_index + 1
        
        result_list = []
        for i in range(len(index_list)):
            start_index, end_index = index_list[i]
            def get_temp_traj(start_index, end_index):
                temp_traj = []
                start_t = self.t_list[start_index]
                
                for i in range(start_index, end_index):
                    min_t = int(np.ceil((self.t_list[i] - start_t) / self.time_unit))
                    max_t = int(np.ceil((self.t_list[i + 1] - start_t) / self.time_unit)) + (1 if i == end_index - 1 else 0)
                    for j in range(min_t, max_t):
                        temp_t = start_t + j * self.time_unit
                        temp = (temp_t - self.t_list[i]) / (self.t_list[i + 1] - self.t_list[i]) * (self.traj[i + 1] - self.traj[i]) + self.traj[i]
                        temp_traj.append(temp)
                
                temp_traj = np.array(temp_traj)
                return temp_traj
            
            temp_traj = get_temp_traj(start_index, end_index)
            
            temp_block_num = (len(temp_traj) - 2) // block_size + 1
            temp_block_size = (len(temp_traj) - 2) // temp_block_num + 1
            for block_id in range(temp_block_num + 1):
                temp_index = min((block_id) * temp_block_size, len(temp_traj) - 1)
                temp_point = temp_traj[temp_index]
                temp_delta = []
                for dim in range(traj_dim):
                    temp_delta.append(temp_point[dim] - round(temp_point[dim] / max_accuracy_error_dim / 2) * max_accuracy_error_dim * 2)
                    temp_traj[temp_index][dim] = round(temp_point[dim] / max_accuracy_error_dim / 2) * max_accuracy_error_dim * 2
                
            delta = temp_traj[1:] - temp_traj[:-1]
            
            result = [[] for _ in range(traj_dim)]
            
            length_bit = max(0, math.ceil(math.log2(int((temp_block_size - 1) * scale_length) + 1)))
            
            for dim in range(traj_dim):
                for block_id in range(temp_block_num):
                    block_dim = delta[block_id * temp_block_size: min((block_id + 1) * temp_block_size, len(delta)), dim]
                    mean = np.mean(block_dim)
                    block_dim = block_dim - mean
                    
                    dctn_dim = dctn(block_dim)
                    compressed_dim = []
                    index = int((len(dctn_dim) - 1) * scale_length)
                    while int(np.round(dctn_dim[index] * scale)) == 0 and index > 0:
                        index = index - 1
                    
                    for j in range(1, index + 1):
                        compressed_dim.append(int(np.round(dctn_dim[j] * scale)))
                    
                    result[dim].append([compressed_dim, temp_traj[min((block_id + 1) * temp_block_size, len(temp_traj) - 1), dim]])
                    
            result_list.append([self.t_list[start_index], temp_traj[0], temp_block_size, len(delta), result])
        
        
        if max_error is not None:
            decompress_traj0 = []
            t_list0 = []
            for i in range(len(result_list)):
                start_t = result_list[i][0]
                start_point = result_list[i][1]
                temp_block_size = result_list[i][2]
                delta_num = result_list[i][3]
                result = result_list[i][4]
                
                tot_idct = [[] for _ in range(traj_dim)]
                for dim in range(traj_dim):
                    for i in range(len(result[dim])):
                        compressed_dim = result[dim][i][0]
                        mean = (result[dim][i][1] - (result[dim][i - 1][1] if i > 0 else start_point[dim])) / min(temp_block_size, delta_num - temp_block_size * i)
                        compressed_dim = np.pad(compressed_dim, (1, min(temp_block_size, delta_num - temp_block_size * i) - len(compressed_dim) - 1), 'constant')
                        idct_dim = idctn(compressed_dim / scale)
                        idct_dim = idct_dim + mean
                        tot_idct[dim].extend(idct_dim)
                tot_idct = np.array(tot_idct).T
                
                if len(t_list0) > 0 and start_t <= t_list0[-1]:
                    decompress_traj0.pop()
                    t_list0.pop()
        
                
                decompress_traj0.append(start_point)
                t_list0.append(start_t)
                for i in range(len(tot_idct)):
                    decompress_traj0.append(decompress_traj0[-1] + tot_idct[i])
                    t_list0.append(t_list0[-1] + self.time_unit)
            
            decompress_traj1 = []
            t_list1 = []
            for i in range(len(other_points)):
                index = other_points[i]
                t = self.t_list[index]
                point = self.traj[index]
                decompress_traj1.append(point)
                t_list1.append(t)
            
            all_decompress_traj = []
            index0 = 0
            index1 = 0
            for i in range(len(self.t_list)):
                t = self.t_list[i]
                while index0 < len(t_list0) - 1 and t_list0[index0 + 1] - t < -0.1 * self.time_accuracy:
                    index0 = index0 + 1
                while index1 < len(t_list1) and t_list1[index1] - t < -0.1 * self.time_accuracy:
                    index1 = index1 + 1
                    
                if index0 == len(t_list0) - 1:
                    temp_point = decompress_traj0[index0]
                elif index0 < len(t_list0) - 1:
                    temp_point = decompress_traj0[index0] + (decompress_traj0[index0 + 1] - decompress_traj0[index0]) * (t - t_list0[index0]) / (t_list0[index0 + 1] - t_list0[index0])
                
                if index1 != len(t_list1) and abs(t_list1[index1] - t) < 0.1 * self.time_accuracy:
                    all_decompress_traj.append(decompress_traj1[index1])
                else:
                    all_decompress_traj.append(temp_point)
            all_decompress_traj = np.array(all_decompress_traj)
            
            error = np.sqrt(np.sum((self.traj - all_decompress_traj) ** 2, axis=1))
            error_cnt = np.where(error > max_error)[0]
            
            for i in error_cnt:
                error_point = []
                delta_point = self.traj[i] - all_decompress_traj[i]
                for j in range(traj_dim):
                    delta_dim = delta_point[j]
                    value_dim = round(delta_dim / max_error_dim / 2)
                    error_point.append(value_dim)
                    all_decompress_traj[i][j] = all_decompress_traj[i][j] + value_dim * max_error_dim * 2
                    
                assert np.sqrt(np.sum((self.traj[i] - all_decompress_traj[i]) ** 2)) < max_error
                
                error_point.append(self.t_list[i])
                error_points.append(error_point)
                
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(varint2bitarray(len(result_list), utf))
                write_bits.extend(double2bitarray(self.time_accuracy))
                write_bits.extend(float2bitarray(max_error))
                # if len(result_list) != 0:
                    # write_bits.extend(float2bitarray(self.time_unit))
                    # write_bits.extend(float2bitarray(scale_length))
                    # write_bits.extend(varint2bitarray(block_size, utf))
                    
                last_value = [0 for i in range(traj_dim)]
                
                for i in range(len(result_list)):
                    start_t = result_list[i][0]
                    start_point = result_list[i][1]
                    temp_block_size = result_list[i][2]
                    delta_num = result_list[i][3]
                    result = result_list[i][4]
                    
                    length_bit = max(0, math.ceil(math.log2(int((temp_block_size - 1) * scale_length) + 1)))
                    if i == 0:
                        write_bits.extend(varint2bitarray(round(start_t / self.time_accuracy), utf))
                    else:
                        write_bits.extend(varint2bitarray(round((start_t - result_list[i - 1][0] - (result_list[i - 1][3] - 1) * self.time_unit) / self.time_accuracy), utf))
                    write_bits.extend(varint2bitarray(delta_num, utf))
                    
                    
                    for dim in range(traj_dim):
                        write_bits.extend(zigzag2bitarray(round(start_point[dim] / max_accuracy_error_dim / 2) - \
                                                          round(last_value[dim] / max_accuracy_error_dim / 2), utf))
                        last_value[dim] = start_point[dim]
                    for i in range(len(result[dim])):
                        for dim in range(traj_dim):
                            write_bits.extend(zigzag2bitarray(round(result[dim][i][1] / max_accuracy_error_dim / 2) - \
                                                              round(last_value[dim] / max_accuracy_error_dim / 2), utf))
                            last_value[dim] = result[dim][i][1]
                        
                    for dim in range(traj_dim):
                        for i in range(len(result[dim])):
                            write_bits.extend(signint2bitarray(len(result[dim][i][0]), length_bit))
                            for j in range(len(result[dim][i][0])):
                                write_bits.extend(customint2bitarray(result[dim][i][0][j]))
                
                write_bits.extend(varint2bitarray(len(error_points), utf))
                for i in range(len(error_points)):
                    if i == 0:
                        write_bits.extend(varint2bitarray(round(error_points[i][-1] / self.time_accuracy), utf))
                    else:
                        write_bits.extend(varint2bitarray(round((error_points[i][-1] - error_points[i - 1][-1]) / self.time_accuracy), utf))
                    
                    for dim in range(traj_dim):
                        write_bits.extend(customint2bitarray(error_points[i][dim]))
                        
                for i in range(len(other_points)):
                    if i == 0:
                        write_bits.extend(varint2bitarray(round(self.t_list[other_points[i]] / self.time_accuracy), utf))
                    else:
                        write_bits.extend(varint2bitarray(round((self.t_list[other_points[i]] - self.t_list[other_points[i - 1]]) / self.time_accuracy), utf))
                    
                    for dim in range(traj_dim):
                        if i == 0:
                            write_bits.extend(zigzag2bitarray(round(self.traj[other_points[i]][dim] / max_error_dim / 2), utf))
                        else:
                            write_bits.extend(zigzag2bitarray(round(self.traj[other_points[i]][dim] / max_error_dim / 2) - \
                                                              round(self.traj[other_points[i - 1]][dim] / max_error_dim / 2), utf))
                            
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
    
    @profile
    def decompress(self, **kwargs):
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}_block_{kwargs['block_size']}_scale_{kwargs['scale_length']}_epsilon_{kwargs['epsilon']}'
        
        result_list = []
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            num, read_index = bitarray2varint(read_bits, read_index, utf)
            time_accuracy = bitarray2double(read_bits[read_index:read_index+64])
            read_index += 64
            max_error = bitarray2float(read_bits[read_index:read_index+32])
            read_index += 32
            max_accuracy_error = kwargs['accuracy'] * max_error if kwargs['accuracy'] != None else self.accuracy_default * max_error

            other_points = []
            error_points = []
            traj_dim = self.traj.shape[1]
            max_error_dim = max_error / math.sqrt(traj_dim)
            max_accuracy_error_dim = max_accuracy_error / math.sqrt(traj_dim)
            
            if num != 0:
                time_unit = self.time_unit
                scale = self.scale_factor / max_error if kwargs['epsilon'] == None else 0.5 / kwargs['epsilon'] / max_error
                block_size = int(max_error * self.a + self.b) if kwargs['block_size'] == None else kwargs['block_size']
                scale_length = min(1, self.c / math.sqrt(max_error)) if kwargs['scale_length'] == None else kwargs['scale_length']
            
            last_value = [0 for i in range(traj_dim)]
            for i in range(num):
                if len(result_list) == 0:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    start_t = time_num
                else:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    start_t = result_list[-1][0] + time_num + round((result_list[-1][3] - 1) * time_unit / self.time_accuracy)
                    
                delta_num, read_index = bitarray2varint(read_bits, read_index, utf)
                
                temp_block_num = (delta_num - 1) // block_size + 1
                temp_block_size = (delta_num - 1) // temp_block_num + 1
                
                length_bit = max(0, math.ceil(math.log2(int((temp_block_size - 1) * scale_length) + 1)))
                
                result = []
                
                point_list = []
                
                for _ in range(temp_block_num + 1):
                    point = []
                    for dim in range(traj_dim):
                        traj_num, read_index = bitarray2zigzag(read_bits, read_index, utf)
                        point.append(last_value[dim] + traj_num * max_accuracy_error_dim * 2)
                        last_value[dim] = point[-1]
                    point_list.append(point)
                
                
                for dim in range(traj_dim):
                    result_dim = []
                    for block_id in range(temp_block_num):
                        block_dim_num = bitarray2signint(read_bits[read_index:read_index+length_bit], False)
                        read_index += length_bit
                        compressed_dim = []
                        for j in range(block_dim_num):
                            value, read_index = bitarry2customint(read_bits, read_index)
                            compressed_dim.append(value)
                        result_dim.append([compressed_dim, point_list[block_id + 1][dim]])
                    result.append(result_dim)
                
                result_list.append([start_t, point_list[0], temp_block_size, delta_num, result])
                
            for i in range(len(result_list)):
                result_list[i][0] *= time_accuracy
            
            error_points_num, read_index = bitarray2varint(read_bits, read_index, utf)
            for i in range(error_points_num):
                if i == 0:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = time_num
                else:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = error_points[-1][0] + time_num
                point = []
                
                for j in range(traj_dim):
                    value_dim, read_index = bitarry2customint(read_bits, read_index)
                    point.append(value_dim * max_error_dim * 2)
                error_points.append([t, point])
            for i in range(len(error_points)):
                error_points[i][0] *= time_accuracy
            
            # error_points = []
            
            while read_index < len(read_bits) - 8:
                if len(other_points) == 0:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = time_num
                else:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = other_points[-1][0] + time_num
                point = []
                
                for dim in range(traj_dim):
                    if len(other_points) == 0:
                        traj_num, read_index = bitarray2zigzag(read_bits, read_index, utf)
                        value_dim = traj_num * max_error_dim * 2
                    else:
                        traj_num, read_index = bitarray2zigzag(read_bits, read_index, utf)
                        value_dim = other_points[-1][-1][dim] + traj_num * max_error_dim * 2
                    point.append(value_dim)
                other_points.append([t, point])
            for i in range(len(other_points)):
                other_points[i][0] *= time_accuracy
        
        decompress_traj0 = []
        t_list0 = []
        for i in range(len(result_list)):
            start_t = result_list[i][0]
            start_point = result_list[i][1]
            temp_block_size = result_list[i][2]
            delta_num = result_list[i][3]
            result = result_list[i][4]
            
            tot_idct = [[] for _ in range(traj_dim)]
            for dim in range(traj_dim):
                for i in range(len(result[dim])):
                    compressed_dim = result[dim][i][0]
                    mean = (result[dim][i][1] - (result[dim][i - 1][1] if i > 0 else start_point[dim])) / min(temp_block_size, delta_num - temp_block_size * i)
                    compressed_dim = np.pad(compressed_dim, (1, min(temp_block_size, delta_num - temp_block_size * i) - len(compressed_dim) - 1), 'constant')
                    idct_dim = idctn(compressed_dim / scale)
                    idct_dim = idct_dim + mean
                    tot_idct[dim].extend(idct_dim)
            tot_idct = np.array(tot_idct).T
            
            if len(t_list0) > 0 and start_t <= t_list0[-1]:
                decompress_traj0.pop()
                t_list0.pop()
    
            
            decompress_traj0.append(np.array(start_point))
            t_list0.append(start_t)
            for i in range(len(tot_idct)):
                decompress_traj0.append(decompress_traj0[-1] + tot_idct[i])
                t_list0.append(t_list0[-1] + time_unit)
        
        decompress_traj1 = []
        t_list1 = []
        for i in range(len(other_points)):
            t = other_points[i][0]
            point = other_points[i][1]
            decompress_traj1.append(point)
            t_list1.append(t)
        
        self.decompress_traj = []
        if len(t_list0) > 0:
            index0 = 0
            index1 = 0
            index2 = 0
            for i in range(len(self.t_list)):
                t = self.t_list[i]
                while index0 < len(t_list0) - 1 and t_list0[index0 + 1] - t < -0.1 * self.time_accuracy:
                    index0 = index0 + 1
                while index1 < len(t_list1) and t_list1[index1] - t < -0.1 * self.time_accuracy:
                    index1 = index1 + 1
                    
                while index2 < len(error_points) and error_points[index2][0] - t < -0.1 * self.time_accuracy:
                    index2 = index2 + 1
                    
                if index0 == len(t_list0) - 1:
                    temp_point = decompress_traj0[index0]
                elif index0 < len(t_list0) - 1:
                    temp_point = decompress_traj0[index0] + (decompress_traj0[index0 + 1] - decompress_traj0[index0]) * (t - t_list0[index0]) / (t_list0[index0 + 1] - t_list0[index0])
                
                if index1 != len(t_list1) and abs(t_list1[index1] - t) < 0.1 * self.time_accuracy:
                    self.decompress_traj.append(decompress_traj1[index1])
                else:
                    self.decompress_traj.append(temp_point)
                    
                if index2 != len(error_points) and abs(error_points[index2][0] - t) < 0.1 * self.time_accuracy:
                    self.decompress_traj[-1] += np.array(error_points[index2][1])
        else:
            self.decompress_traj = decompress_traj1
        self.decompress_traj = np.array(self.decompress_traj)

class PILOTCNOENHANCEDZIGZAGCompressor(Compressor):
    def __init__(self, path, data_source='nuplan'):
        super().__init__('PILOT-C-No-Enhanced-Zigzag', path, data_source)
        self.scale_factor = 0.3
        if self.data_source == 'geolife':
            self.utf_default = 3
            self.error_default = 50
            self.a = 0.5
            self.b = 25
            self.c = 1.1
            self.time_unit = 3
        if self.data_source == 'geolife_3d':
            self.utf_default = 3
            self.error_default = 50
            self.a = 0.5
            self.b = 25
            self.c = 0.8
            self.time_unit = 3
            self.scale_factor = 0.35
        if self.data_source == 'mopsi':
            self.utf_default = 3
            self.error_default = 50
            self.a = 1
            self.b = 25
            self.c = 0.6
            self.time_unit = 2
        if self.data_source == 'nuplan':
            self.utf_default = 3
            self.error_default = 5
            self.a = 20
            self.b = 100
            self.c = 0.04
            self.time_unit = 0.1
        self.accuracy_default = 0.5
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        
        max_error = bitarray2float(float2bitarray(max_error))
        block_size = int(max_error * self.a + self.b) if kwargs['block_size'] == None else kwargs['block_size']
        scale_length = min(1, self.c / math.sqrt(max_error)) if kwargs['scale_length'] == None else kwargs['scale_length']
        max_accuracy_error = accuracy * max_error
        
        self.scale_length = scale_length
        
        scale = self.scale_factor / max_error
        max_error_dim = max_error / math.sqrt(traj_dim)
        max_accuracy_error_dim = max_accuracy_error / math.sqrt(traj_dim)
        
        other_points = []
        error_points = []
        index_list = []
        
        start_index = 0
        end_index = 0
        while start_index < len(self.t_list):
            end_index = start_index
            while end_index < len(self.t_list) - 1 and \
                  self.t_list[end_index + 1] - self.t_list[end_index] < block_size * ((self.t_list[end_index] - self.t_list[start_index]) / (end_index - start_index + 1) if end_index != start_index else 1) and \
                  np.sqrt(np.sum((self.traj[end_index + 1] - self.traj[end_index]) ** 2) / (self.t_list[end_index + 1] - self.t_list[end_index])) < 200:
                end_index = end_index + 1
            
            if end_index - start_index <= 1:
                for i in range(start_index, end_index + 1):
                    other_points.append(i)
                start_index = end_index + 1
                continue
            index_list.append([start_index, end_index])
            start_index = end_index + 1
        
        result_list = []
        for i in range(len(index_list)):
            start_index, end_index = index_list[i]
            def get_temp_traj(start_index, end_index):
                temp_traj = []
                start_t = self.t_list[start_index]
                
                for i in range(start_index, end_index):
                    min_t = int(np.ceil((self.t_list[i] - start_t) / self.time_unit))
                    max_t = int(np.ceil((self.t_list[i + 1] - start_t) / self.time_unit)) + (1 if i == end_index - 1 else 0)
                    for j in range(min_t, max_t):
                        temp_t = start_t + j * self.time_unit
                        temp = (temp_t - self.t_list[i]) / (self.t_list[i + 1] - self.t_list[i]) * (self.traj[i + 1] - self.traj[i]) + self.traj[i]
                        temp_traj.append(temp)
                
                temp_traj = np.array(temp_traj)
                return temp_traj
            
            temp_traj = get_temp_traj(start_index, end_index)
            
            temp_block_num = (len(temp_traj) - 2) // block_size + 1
            temp_block_size = (len(temp_traj) - 2) // temp_block_num + 1
            for block_id in range(temp_block_num + 1):
                temp_index = min((block_id) * temp_block_size, len(temp_traj) - 1)
                temp_point = temp_traj[temp_index]
                temp_delta = []
                for dim in range(traj_dim):
                    temp_delta.append(temp_point[dim] - round(temp_point[dim] / max_accuracy_error_dim / 2) * max_accuracy_error_dim * 2)
                    temp_traj[temp_index][dim] = round(temp_point[dim] / max_accuracy_error_dim / 2) * max_accuracy_error_dim * 2
                
            delta = temp_traj[1:] - temp_traj[:-1]
            
            result = [[] for _ in range(traj_dim)]
            
            length_bit = max(0, math.ceil(math.log2(int((temp_block_size - 1) * scale_length) + 1)))
            
            for dim in range(traj_dim):
                for block_id in range(temp_block_num):
                    block_dim = delta[block_id * temp_block_size: min((block_id + 1) * temp_block_size, len(delta)), dim]
                    mean = np.mean(block_dim)
                    block_dim = block_dim - mean
                    
                    dctn_dim = dctn(block_dim)
                    compressed_dim = []
                    index = int((len(dctn_dim) - 1) * scale_length)
                    while int(np.round(dctn_dim[index] * scale)) == 0 and index > 0:
                        index = index - 1
                    
                    for j in range(1, index + 1):
                        compressed_dim.append(int(np.round(dctn_dim[j] * scale)))
                    
                    result[dim].append([compressed_dim, temp_traj[min((block_id + 1) * temp_block_size, len(temp_traj) - 1), dim]])
                    
            result_list.append([self.t_list[start_index], temp_traj[0], temp_block_size, len(delta), result])
        
        
        if max_error is not None:
            decompress_traj0 = []
            t_list0 = []
            for i in range(len(result_list)):
                start_t = result_list[i][0]
                start_point = result_list[i][1]
                temp_block_size = result_list[i][2]
                delta_num = result_list[i][3]
                result = result_list[i][4]
                
                tot_idct = [[] for _ in range(traj_dim)]
                for dim in range(traj_dim):
                    for i in range(len(result[dim])):
                        compressed_dim = result[dim][i][0]
                        mean = (result[dim][i][1] - (result[dim][i - 1][1] if i > 0 else start_point[dim])) / min(temp_block_size, delta_num - temp_block_size * i)
                        compressed_dim = np.pad(compressed_dim, (1, min(temp_block_size, delta_num - temp_block_size * i) - len(compressed_dim) - 1), 'constant')
                        idct_dim = idctn(compressed_dim / scale)
                        idct_dim = idct_dim + mean
                        tot_idct[dim].extend(idct_dim)
                tot_idct = np.array(tot_idct).T
                
                if len(t_list0) > 0 and start_t <= t_list0[-1]:
                    decompress_traj0.pop()
                    t_list0.pop()
        
                
                decompress_traj0.append(start_point)
                t_list0.append(start_t)
                for i in range(len(tot_idct)):
                    decompress_traj0.append(decompress_traj0[-1] + tot_idct[i])
                    t_list0.append(t_list0[-1] + self.time_unit)
            
            decompress_traj1 = []
            t_list1 = []
            for i in range(len(other_points)):
                index = other_points[i]
                t = self.t_list[index]
                point = self.traj[index]
                decompress_traj1.append(point)
                t_list1.append(t)
            
            all_decompress_traj = []
            index0 = 0
            index1 = 0
            for i in range(len(self.t_list)):
                t = self.t_list[i]
                while index0 < len(t_list0) - 1 and t_list0[index0 + 1] - t < -0.1 * self.time_accuracy:
                    index0 = index0 + 1
                while index1 < len(t_list1) and t_list1[index1] - t < -0.1 * self.time_accuracy:
                    index1 = index1 + 1
                    
                if index0 == len(t_list0) - 1:
                    temp_point = decompress_traj0[index0]
                elif index0 < len(t_list0) - 1:
                    temp_point = decompress_traj0[index0] + (decompress_traj0[index0 + 1] - decompress_traj0[index0]) * (t - t_list0[index0]) / (t_list0[index0 + 1] - t_list0[index0])
                
                if index1 != len(t_list1) and abs(t_list1[index1] - t) < 0.1 * self.time_accuracy:
                    all_decompress_traj.append(decompress_traj1[index1])
                else:
                    all_decompress_traj.append(temp_point)
            all_decompress_traj = np.array(all_decompress_traj)
            
            error = np.sqrt(np.sum((self.traj - all_decompress_traj) ** 2, axis=1))
            error_cnt = np.where(error > max_error)[0]
            
            for i in error_cnt:
                error_point = []
                delta_point = self.traj[i] - all_decompress_traj[i]
                for j in range(traj_dim):
                    delta_dim = delta_point[j]
                    value_dim = round(delta_dim / max_error_dim / 2)
                    error_point.append(value_dim)
                    all_decompress_traj[i][j] = all_decompress_traj[i][j] + value_dim * max_error_dim * 2
                    
                assert np.sqrt(np.sum((self.traj[i] - all_decompress_traj[i]) ** 2)) < max_error
                
                error_point.append(self.t_list[i])
                error_points.append(error_point)
                
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(varint2bitarray(len(result_list), utf))
                write_bits.extend(double2bitarray(self.time_accuracy))
                write_bits.extend(float2bitarray(max_error))
                # if len(result_list) != 0:
                    # write_bits.extend(float2bitarray(self.time_unit))
                    # write_bits.extend(float2bitarray(scale_length))
                    # write_bits.extend(varint2bitarray(block_size, utf))
                    
                last_value = [0 for i in range(traj_dim)]
                
                for i in range(len(result_list)):
                    start_t = result_list[i][0]
                    start_point = result_list[i][1]
                    temp_block_size = result_list[i][2]
                    delta_num = result_list[i][3]
                    result = result_list[i][4]
                    
                    length_bit = max(0, math.ceil(math.log2(int((temp_block_size - 1) * scale_length) + 1)))
                    if i == 0:
                        write_bits.extend(varint2bitarray(round(start_t / self.time_accuracy), utf))
                    else:
                        write_bits.extend(varint2bitarray(round((start_t - result_list[i - 1][0] - (result_list[i - 1][3] - 1) * self.time_unit) / self.time_accuracy), utf))
                    write_bits.extend(varint2bitarray(delta_num, utf))
                    
                    
                    for dim in range(traj_dim):
                        write_bits.extend(zigzag2bitarray(round(start_point[dim] / max_accuracy_error_dim / 2) - \
                                                          round(last_value[dim] / max_accuracy_error_dim / 2), utf))
                        last_value[dim] = start_point[dim]
                    for i in range(len(result[dim])):
                        for dim in range(traj_dim):
                            write_bits.extend(zigzag2bitarray(round(result[dim][i][1] / max_accuracy_error_dim / 2) - \
                                                              round(last_value[dim] / max_accuracy_error_dim / 2), utf))
                            last_value[dim] = result[dim][i][1]
                        
                    for dim in range(traj_dim):
                        for i in range(len(result[dim])):
                            write_bits.extend(signint2bitarray(len(result[dim][i][0]), length_bit))
                            for j in range(len(result[dim][i][0])):
                                write_bits.extend(zigzag2bitarray(result[dim][i][0][j], utf))
                
                write_bits.extend(varint2bitarray(len(error_points), utf))
                for i in range(len(error_points)):
                    if i == 0:
                        write_bits.extend(varint2bitarray(round(error_points[i][-1] / self.time_accuracy), utf))
                    else:
                        write_bits.extend(varint2bitarray(round((error_points[i][-1] - error_points[i - 1][-1]) / self.time_accuracy), utf))
                    
                    for dim in range(traj_dim):
                        write_bits.extend(zigzag2bitarray(error_points[i][dim], utf))
                        
                for i in range(len(other_points)):
                    if i == 0:
                        write_bits.extend(varint2bitarray(round(self.t_list[other_points[i]] / self.time_accuracy), utf))
                    else:
                        write_bits.extend(varint2bitarray(round((self.t_list[other_points[i]] - self.t_list[other_points[i - 1]]) / self.time_accuracy), utf))
                    
                    for dim in range(traj_dim):
                        if i == 0:
                            write_bits.extend(zigzag2bitarray(round(self.traj[other_points[i]][dim] / max_error_dim / 2), utf))
                        else:
                            write_bits.extend(zigzag2bitarray(round(self.traj[other_points[i]][dim] / max_error_dim / 2) - \
                                                              round(self.traj[other_points[i - 1]][dim] / max_error_dim / 2), utf))
                            
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
    
    @profile
    def decompress(self, **kwargs):
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        
        result_list = []
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            num, read_index = bitarray2varint(read_bits, read_index, utf)
            time_accuracy = bitarray2double(read_bits[read_index:read_index+64])
            read_index += 64
            max_error = bitarray2float(read_bits[read_index:read_index+32])
            read_index += 32
            max_accuracy_error = kwargs['accuracy'] * max_error if kwargs['accuracy'] != None else self.accuracy_default * max_error

            other_points = []
            error_points = []
            traj_dim = self.traj.shape[1]
            max_error_dim = max_error / math.sqrt(traj_dim)
            max_accuracy_error_dim = max_accuracy_error / math.sqrt(traj_dim)
            
            if num != 0:
                time_unit = self.time_unit
                scale = self.scale_factor / max_error
                block_size = int(max_error * self.a + self.b) if kwargs['block_size'] == None else kwargs['block_size']
                scale_length = min(1, self.c / math.sqrt(max_error)) if kwargs['scale_length'] == None else kwargs['scale_length']
            
            last_value = [0 for i in range(traj_dim)]
            for i in range(num):
                if len(result_list) == 0:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    start_t = time_num
                else:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    start_t = result_list[-1][0] + time_num + round((result_list[-1][3] - 1) * time_unit / self.time_accuracy)
                    
                delta_num, read_index = bitarray2varint(read_bits, read_index, utf)
                
                temp_block_num = (delta_num - 1) // block_size + 1
                temp_block_size = (delta_num - 1) // temp_block_num + 1
                
                length_bit = max(0, math.ceil(math.log2(int((temp_block_size - 1) * scale_length) + 1)))
                
                result = []
                
                point_list = []
                
                for _ in range(temp_block_num + 1):
                    point = []
                    for dim in range(traj_dim):
                        traj_num, read_index = bitarray2zigzag(read_bits, read_index, utf)
                        point.append(last_value[dim] + traj_num * max_accuracy_error_dim * 2)
                        last_value[dim] = point[-1]
                    point_list.append(point)
                
                
                for dim in range(traj_dim):
                    result_dim = []
                    for block_id in range(temp_block_num):
                        block_dim_num = bitarray2signint(read_bits[read_index:read_index+length_bit], False)
                        read_index += length_bit
                        compressed_dim = []
                        for j in range(block_dim_num):
                            value, read_index = bitarray2zigzag(read_bits, read_index, utf)
                            compressed_dim.append(value)
                        result_dim.append([compressed_dim, point_list[block_id + 1][dim]])
                    result.append(result_dim)
                
                result_list.append([start_t, point_list[0], temp_block_size, delta_num, result])
                
            for i in range(len(result_list)):
                result_list[i][0] *= time_accuracy
            
            error_points_num, read_index = bitarray2varint(read_bits, read_index, utf)
            for i in range(error_points_num):
                if i == 0:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = time_num
                else:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = error_points[-1][0] + time_num
                point = []
                
                for j in range(traj_dim):
                    value_dim, read_index = bitarray2zigzag(read_bits, read_index, utf)
                    point.append(value_dim * max_error_dim * 2)
                error_points.append([t, point])
            for i in range(len(error_points)):
                error_points[i][0] *= time_accuracy
            
            while read_index < len(read_bits) - 8:
                if len(other_points) == 0:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = time_num
                else:
                    time_num, read_index = bitarray2varint(read_bits, read_index, utf)
                    t = other_points[-1][0] + time_num
                point = []
                
                for dim in range(traj_dim):
                    if len(other_points) == 0:
                        traj_num, read_index = bitarray2zigzag(read_bits, read_index, utf)
                        value_dim = traj_num * max_error_dim * 2
                    else:
                        traj_num, read_index = bitarray2zigzag(read_bits, read_index, utf)
                        value_dim = other_points[-1][-1][dim] + traj_num * max_error_dim * 2
                    point.append(value_dim)
                other_points.append([t, point])
            for i in range(len(other_points)):
                other_points[i][0] *= time_accuracy
        
        decompress_traj0 = []
        t_list0 = []
        for i in range(len(result_list)):
            start_t = result_list[i][0]
            start_point = result_list[i][1]
            temp_block_size = result_list[i][2]
            delta_num = result_list[i][3]
            result = result_list[i][4]
            
            tot_idct = [[] for _ in range(traj_dim)]
            for dim in range(traj_dim):
                for i in range(len(result[dim])):
                    compressed_dim = result[dim][i][0]
                    mean = (result[dim][i][1] - (result[dim][i - 1][1] if i > 0 else start_point[dim])) / min(temp_block_size, delta_num - temp_block_size * i)
                    compressed_dim = np.pad(compressed_dim, (1, min(temp_block_size, delta_num - temp_block_size * i) - len(compressed_dim) - 1), 'constant')
                    idct_dim = idctn(compressed_dim / scale)
                    idct_dim = idct_dim + mean
                    tot_idct[dim].extend(idct_dim)
            tot_idct = np.array(tot_idct).T
            
            if len(t_list0) > 0 and start_t <= t_list0[-1]:
                decompress_traj0.pop()
                t_list0.pop()
    
            
            decompress_traj0.append(np.array(start_point))
            t_list0.append(start_t)
            for i in range(len(tot_idct)):
                decompress_traj0.append(decompress_traj0[-1] + tot_idct[i])
                t_list0.append(t_list0[-1] + time_unit)
        
        decompress_traj1 = []
        t_list1 = []
        for i in range(len(other_points)):
            t = other_points[i][0]
            point = other_points[i][1]
            decompress_traj1.append(point)
            t_list1.append(t)
        
        self.decompress_traj = []
        if len(t_list0) > 0:
            index0 = 0
            index1 = 0
            index2 = 0
            for i in range(len(self.t_list)):
                t = self.t_list[i]
                while index0 < len(t_list0) - 1 and t_list0[index0 + 1] - t < -0.1 * self.time_accuracy:
                    index0 = index0 + 1
                while index1 < len(t_list1) and t_list1[index1] - t < -0.1 * self.time_accuracy:
                    index1 = index1 + 1
                    
                while index2 < len(error_points) and error_points[index2][0] - t < -0.1 * self.time_accuracy:
                    index2 = index2 + 1
                    
                if index0 == len(t_list0) - 1:
                    temp_point = decompress_traj0[index0]
                elif index0 < len(t_list0) - 1:
                    temp_point = decompress_traj0[index0] + (decompress_traj0[index0 + 1] - decompress_traj0[index0]) * (t - t_list0[index0]) / (t_list0[index0 + 1] - t_list0[index0])
                
                if index1 != len(t_list1) and abs(t_list1[index1] - t) < 0.1 * self.time_accuracy:
                    self.decompress_traj.append(decompress_traj1[index1])
                else:
                    self.decompress_traj.append(temp_point)
                    
                if index2 != len(error_points) and abs(error_points[index2][0] - t) < 0.1 * self.time_accuracy:
                    self.decompress_traj[-1] += np.array(error_points[index2][1])
        else:
            self.decompress_traj = decompress_traj1
        self.decompress_traj = np.array(self.decompress_traj)

class CISEDSCompressor(Compressor):
    def __init__(self, path, data_source):
        super().__init__('CISED-S', path, data_source)
        if self.data_source == 'geolife':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'geolife_3d':
            assert False, 'Not implemented'
        if self.data_source == 'mopsi':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'nuplan':
            self.utf_default = 3
            self.error_default = 5
        self.accuracy_default = 0.2
        self.polygon_num_default = 16
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        max_accuracy_error = accuracy * max_error
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        polygon_num = kwargs['polygon_num'] if 'polygon_num' in kwargs else self.polygon_num_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        
        max_accuracy_error = bitarray2float(float2bitarray(max_accuracy_error))
        max_error_dim = max_accuracy_error / math.sqrt(2)
        
        max_error = max_error - max_accuracy_error
        
        from shapely.geometry import Polygon
        
        index_list = [0]
        polygon = Polygon()
        for i in range(1, len(self.traj)):
            def get_polygon(index0, index1, r):
                c = (self.t_list[index0 + 1] - self.t_list[index0]) / (self.t_list[index1] - self.t_list[index0])
                
                point = self.traj[index0] + (self.traj[index1] - self.traj[index0]) * c
                point_list = []
                for j in range(polygon_num):
                    angle = (2 * j - 1) * np.pi / polygon_num
                    point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) * c)
                return Polygon(point_list)
            
            temp_polygon = get_polygon(index_list[-1], i, max_error / 2)
            if polygon.is_empty:
                polygon = temp_polygon
            else:
                polygon = polygon.intersection(temp_polygon)
                if polygon.is_empty:
                    index_list.append(i - 1)
                    polygon = get_polygon(index_list[-1], i, max_error / 2)
        index_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(double2bitarray(self.time_accuracy))
                write_bits.extend(float2bitarray(max_accuracy_error))
                
                for i in range(0, len(index_list)):
                    temp_t = round((self.t_list[index_list[i]] - self.t_list[index_list[i - 1]]) / self.time_accuracy) if i != 0 else round(self.t_list[index_list[i]] / self.time_accuracy)
                    write_bits.extend(varint2bitarray(temp_t, utf))
                    
                    for dim in range(traj_dim):
                        temp_dim = round(self.traj[index_list[i]][dim] / max_error_dim / 2) - round(self.traj[index_list[i - 1]][dim] / max_error_dim / 2) \
                                   if i != 0 else round(self.traj[index_list[i]][dim] / max_error_dim / 2)
                        write_bits.extend(zigzag2bitarray(temp_dim, utf))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, **kwargs):
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        traj_dim = self.traj.shape[1]
        
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            t_list = []
            point_list = []
            time_accuracy = bitarray2double(read_bits[read_index:read_index+64])
            read_index += 64
            max_accuracy_error = bitarray2float(read_bits[read_index:read_index+32])
            read_index += 32
            max_error_dim = max_accuracy_error / math.sqrt(2)
            
            while read_index < len(read_bits) - 8:
                t, read_index = bitarray2varint(read_bits, read_index, utf)
                point = []
                for dim in range(traj_dim):
                    temp, read_index = bitarray2zigzag(read_bits, read_index, utf)
                    point.append(temp)
                if len(t_list) == 0:
                    t_list.append(t)
                    point_list.append(np.array(point) * max_error_dim * 2)
                else:
                    t_list.append(t + t_list[-1])
                    point_list.append(np.array(point) * max_error_dim * 2 + point_list[-1])
            
            for i in range(len(t_list)):
                t_list[i] = t_list[i] * time_accuracy
        
        t_list = np.array(t_list)
        point_list = np.array(point_list)
        
        self.decompress_traj = []
        index = 0
        for i in range(len(self.t_list)):
            t = self.t_list[i]
            while index < len(t_list) - 1 and t_list[index + 1] - t < -1e-4:
                index = index + 1
            if index == len(t_list) - 1:
                temp_point = point_list[index]
            else:
                temp_point = point_list[index] + (point_list[index + 1] - point_list[index]) * (t - t_list[index]) / (t_list[index + 1] - t_list[index])
            
            self.decompress_traj.append(temp_point)
            
        self.decompress_traj = np.array(self.decompress_traj)
        
class CISEDWCompressor(Compressor):
    def __init__(self, path, data_source):
        super().__init__('CISED-W', path, data_source)
        if self.data_source == 'geolife':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'geolife_3d':
            assert False, 'Not implemented'
        if self.data_source == 'mopsi':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'nuplan':
            self.utf_default = 3
            self.error_default = 5
        self.accuracy_default = 0.2
        self.polygon_num_default = 16
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        max_accuracy_error = accuracy * max_error
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        polygon_num = kwargs['polygon_num'] if 'polygon_num' in kwargs else self.polygon_num_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        
        max_accuracy_error = bitarray2float(float2bitarray(max_accuracy_error))
        max_error_dim = max_accuracy_error / math.sqrt(2)
        
        max_error = max_error - max_accuracy_error
        
        from shapely.geometry import Polygon, Point
        
        point_list = [self.traj[0]]
        index_list = [0]
        polygon = Polygon()
        for i in range(1, len(self.traj)):
            def get_polygon(point0, point1, index0, index1, r):
                c = (self.t_list[index0 + 1] - self.t_list[index0]) / (self.t_list[index1] - self.t_list[index0])
                point = point0 + (point1 - point0) * c
                point_list = []
                for j in range(polygon_num):
                    angle = (2 * j - 1) * np.pi / polygon_num
                    point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) * c)
                return Polygon(point_list)
            
            temp_polygon = get_polygon(point_list[-1], self.traj[i], index_list[-1], i, max_error)
            if polygon.is_empty:
                polygon = temp_polygon
            else:
                center_point = np.array([polygon.centroid.x, polygon.centroid.y])
                new_polygon = polygon.intersection(temp_polygon)
                if new_polygon.is_empty:
                    c = (self.t_list[index_list[-1] + 1] - self.t_list[index_list[-1]]) / (self.t_list[i - 1] - self.t_list[index_list[-1]])
                    add_point = self.traj[i - 1]
                    temp_point = point_list[-1] + (add_point - point_list[-1]) * c
                    if not polygon.contains(Point(temp_point)):
                        add_point = point_list[-1] + (center_point - point_list[-1]) / c
                    
                    point_list.append(add_point)
                    index_list.append(i - 1)
                    polygon = get_polygon(add_point, self.traj[i], index_list[-1], i, max_error)
                else:
                    polygon = new_polygon
        
        add_point = self.traj[-1]
        if not polygon.is_empty:
            center_point = np.array([polygon.centroid.x, polygon.centroid.y])
            c = (self.t_list[index_list[-1] + 1] - self.t_list[index_list[-1]]) / (self.t_list[len(self.traj) - 1] - self.t_list[index_list[-1]])
            temp_point = point_list[-1] + (add_point - point_list[-1]) * c
            if not polygon.contains(Point(temp_point)):
                add_point = point_list[-1] + (center_point - point_list[-1]) / c
        
        point_list.append(add_point)
        index_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(double2bitarray(self.time_accuracy))
                write_bits.extend(float2bitarray(max_accuracy_error))
                
                for i in range(0, len(index_list)):
                    temp_t = round((self.t_list[index_list[i]] - self.t_list[index_list[i - 1]]) / self.time_accuracy) if i != 0 else round(self.t_list[index_list[i]] / self.time_accuracy)
                    write_bits.extend(varint2bitarray(temp_t, utf))
                    
                    for dim in range(traj_dim):
                        temp_dim = round(point_list[i][dim] / max_error_dim / 2) - round(point_list[i - 1][dim] / max_error_dim / 2) \
                                   if i != 0 else round(point_list[i][dim] / max_error_dim / 2)
                        write_bits.extend(zigzag2bitarray(temp_dim, utf))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
    
    def decompress(self, **kwargs):
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        traj_dim = self.traj.shape[1]
        
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            t_list = []
            point_list = []
            time_accuracy = bitarray2double(read_bits[read_index:read_index+64])
            read_index += 64
            max_accuracy_error = bitarray2float(read_bits[read_index:read_index+32])
            read_index += 32
            max_error_dim = max_accuracy_error / math.sqrt(2)
            
            while read_index < len(read_bits) - 8:
                t, read_index = bitarray2varint(read_bits, read_index, utf)
                point = []
                for dim in range(traj_dim):
                    temp, read_index = bitarray2zigzag(read_bits, read_index, utf)
                    point.append(temp)
                if len(t_list) == 0:
                    t_list.append(t)
                    point_list.append(np.array(point) * max_error_dim * 2)
                else:
                    t_list.append(t + t_list[-1])
                    point_list.append(np.array(point) * max_error_dim * 2 + point_list[-1])
            
            for i in range(len(t_list)):
                t_list[i] = t_list[i] * time_accuracy
        
        t_list = np.array(t_list)
        point_list = np.array(point_list)
        
        self.decompress_traj = []
        index = 0
        for i in range(len(self.t_list)):
            t = self.t_list[i]
            while index < len(t_list) - 1 and t_list[index + 1] - t < -0.1 * self.time_accuracy:
                index = index + 1
            if index == len(t_list) - 1:
                temp_point = point_list[index]
            else:
                temp_point = point_list[index] + (point_list[index + 1] - point_list[index]) * (t - t_list[index]) / (t_list[index + 1] - t_list[index])
            
            self.decompress_traj.append(temp_point)
            
        self.decompress_traj = np.array(self.decompress_traj)
        
class SQUISHCompressor(Compressor):
    def __init__(self, path, data_source):
        super().__init__('SQUISH-E', path, data_source)
        if self.data_source == 'geolife':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'geolife_3d':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'mopsi':
            self.utf_default = 3
            self.error_default = 50
        if self.data_source == 'nuplan':
            self.utf_default = 3
            self.error_default = 5
        self.accuracy_default = 0.2
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        max_accuracy_error = accuracy * max_error
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        compression_ratio = kwargs['compression_ratio'] if 'compression_ratio' in kwargs else 1
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        
        max_accuracy_error = bitarray2float(float2bitarray(max_accuracy_error))
        max_error_dim = max_accuracy_error / math.sqrt(2)
        
        max_error = max_error - max_accuracy_error
        
        capacity = len(self.traj) / compression_ratio
        heap = []
        
        pi = np.zeros(len(self.traj))
        succ = np.zeros(len(self.traj), dtype=int) - 1
        pred = np.zeros(len(self.traj), dtype=int) - 1
        
        dict = {}
        
        for i in range(len(self.traj)):
            def adjust(j):
                if pred[j] != -1 and succ[j] != -1:
                    temp_point = self.traj[pred[j]] + (self.traj[succ[j]] - self.traj[pred[j]]) / (self.t_list[succ[j]] - self.t_list[pred[j]]) * (self.t_list[j] - self.t_list[pred[j]])
                    temp = pi[j] + np.linalg.norm(self.traj[j] - temp_point)
                    heapq.heappush(heap, (temp, j))
                    dict[j] = temp
            
            def reduce():
                data = heapq.heappop(heap)
                p, j = data[0], data[1]
                while j in dict and dict[j] != p:
                    data = heapq.heappop(heap)
                    p, j = data[0], data[1]
                
                pi[succ[j]] = max(p, pi[succ[j]])
                pi[pred[j]] = max(p, pi[pred[j]])
                succ[pred[j]] = succ[j]
                pred[succ[j]] = pred[j]
                adjust(pred[j])
                adjust(succ[j])
                
            
            heapq.heappush(heap, (np.inf, i))
            pi[i] = 0
            if i > 0:
                succ[i - 1] = i
                pred[i] = i - 1
                adjust(i - 1)
            if i + 1 > capacity:
                reduce()
                
        data = heapq.heappop(heap)
        p, j = data[0], data[1]
        while j in dict and dict[j] != p:
            data = heapq.heappop(heap)
            p, j = data[0], data[1]
        while p < max_error:
            pi[succ[j]] = max(p, pi[succ[j]])
            pi[pred[j]] = max(p, pi[pred[j]])
            succ[pred[j]] = succ[j]
            pred[succ[j]] = pred[j]
            adjust(pred[j])
            adjust(succ[j])
            
            data = heapq.heappop(heap)
            p, j = data[0], data[1]
            while j in dict and dict[j] != p:
                data = heapq.heappop(heap)
                p, j = data[0], data[1]
    
        index_list = [0]
        while index_list[-1] != len(self.traj) - 1:
            index_list.append(succ[index_list[-1]])
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(double2bitarray(self.time_accuracy))
                write_bits.extend(float2bitarray(max_accuracy_error))
                
                for i in range(0, len(index_list)):
                    temp_t = round((self.t_list[index_list[i]] - self.t_list[index_list[i - 1]]) / self.time_accuracy) if i != 0 else round(self.t_list[index_list[i]] / self.time_accuracy)
                    write_bits.extend(varint2bitarray(temp_t, utf))
                    
                    for dim in range(traj_dim):
                        temp_dim = round(self.traj[index_list[i]][dim] / max_error_dim / 2) - round(self.traj[index_list[i - 1]][dim] / max_error_dim / 2) \
                                   if i != 0 else round(self.traj[index_list[i]][dim] / max_error_dim / 2)
                        write_bits.extend(zigzag2bitarray(temp_dim, utf))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
    
    def decompress(self, **kwargs):
        utf = kwargs['utf'] if kwargs['utf'] != None else self.utf_default
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        accuracy = kwargs['accuracy'] if kwargs['accuracy'] != None else self.accuracy_default
        save_path = self.save_path + f'_utf_{utf}_accuracy_{accuracy}_error_{max_error}'
        traj_dim = self.traj.shape[1]
        
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            t_list = []
            point_list = []
            time_accuracy = bitarray2double(read_bits[read_index:read_index+64])
            read_index += 64
            max_accuracy_error = bitarray2float(read_bits[read_index:read_index+32])
            read_index += 32
            max_error_dim = max_accuracy_error / math.sqrt(2)
            
            while read_index < len(read_bits) - 8:
                t, read_index = bitarray2varint(read_bits, read_index, utf)
                point = []
                for dim in range(traj_dim):
                    temp, read_index = bitarray2zigzag(read_bits, read_index, utf)
                    point.append(temp)
                if len(t_list) == 0:
                    t_list.append(t)
                    point_list.append(np.array(point) * max_error_dim * 2)
                else:
                    t_list.append(t + t_list[-1])
                    point_list.append(np.array(point) * max_error_dim * 2 + point_list[-1])
            
            for i in range(len(t_list)):
                t_list[i] = t_list[i] * time_accuracy
        
        t_list = np.array(t_list)
        point_list = np.array(point_list)
        
        self.decompress_traj = []
        index = 0
        for i in range(len(self.t_list)):
            t = self.t_list[i]
            while index < len(t_list) - 1 and t_list[index + 1] - t < 1e-4:
                index = index + 1
            if index == len(t_list) - 1:
                temp_point = point_list[index]
            else:
                temp_point = point_list[index] + (point_list[index + 1] - point_list[index]) * (t - t_list[index]) / (t_list[index + 1] - t_list[index])
            
            self.decompress_traj.append(temp_point)
            
        self.decompress_traj = np.array(self.decompress_traj)

class CISEDSNOENCODECompressor(Compressor):
    def __init__(self, path, data_source):
        super().__init__('CISED-S-No-Encode', path, data_source)
        if self.data_source == 'geolife':
            self.error_default = 50
        if self.data_source == 'geolife_3d':
            self.error_default = 50
        if self.data_source == 'mopsi':
            self.error_default = 50
        if self.data_source == 'nuplan':
            self.error_default = 5
        self.polygon_num_default = 16
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        polygon_num = kwargs['polygon_num'] if 'polygon_num' in kwargs else self.polygon_num_default
        save_path = self.save_path + f'_error_{max_error}'
        
        from shapely.geometry import Polygon
        
        index_list = [0]
        polygon = Polygon()
        for i in range(1, len(self.traj)):
            def get_polygon(index0, index1, r):
                c = (self.t_list[index0 + 1] - self.t_list[index0]) / (self.t_list[index1] - self.t_list[index0])
                
                point = self.traj[index0] + (self.traj[index1] - self.traj[index0]) * c
                point_list = []
                for j in range(polygon_num):
                    angle = (2 * j - 1) * np.pi / polygon_num
                    point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) * c)
                return Polygon(point_list)
            
            temp_polygon = get_polygon(index_list[-1], i, max_error / 2)
            if polygon.is_empty:
                polygon = temp_polygon
            else:
                polygon = polygon.intersection(temp_polygon)
                if polygon.is_empty:
                    index_list.append(i - 1)
                    polygon = get_polygon(index_list[-1], i, max_error / 2)
        index_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                for i in range(0, len(index_list)):
                    write_bits.extend(double2bitarray(self.t_list[index_list[i]]))
                    for dim in range(traj_dim):
                        write_bits.extend(double2bitarray(self.traj[index_list[i]][dim]))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, **kwargs):
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        save_path = self.save_path + f'_error_{max_error}'
        traj_dim = self.traj.shape[1]
        
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            t_list = []
            point_list = []
            while read_index < len(read_bits) - 8:
                t = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                point = []
                for dim in range(traj_dim):
                    temp = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    point.append(temp)
                t_list.append(t)
                point_list.append(np.array(point))
        
        t_list = np.array(t_list)
        point_list = np.array(point_list)
        
        self.decompress_traj = []
        index = 0
        for i in range(len(self.t_list)):
            t = self.t_list[i]
            while index < len(t_list) - 1 and t_list[index + 1] - t < -1e-4:
                index = index + 1
            if index == len(t_list) - 1:
                temp_point = point_list[index]
            else:
                temp_point = point_list[index] + (point_list[index + 1] - point_list[index]) * (t - t_list[index]) / (t_list[index + 1] - t_list[index])
            
            self.decompress_traj.append(temp_point)
            
        self.decompress_traj = np.array(self.decompress_traj)
        
class CISEDWNOENCODECompressor(Compressor):
    def __init__(self, path, data_source):
        super().__init__('CISED-W-No-Encode', path, data_source)
        if self.data_source == 'geolife':
            self.error_default = 50
        if self.data_source == 'geolife_3d':
            self.error_default = 50
        if self.data_source == 'mopsi':
            self.error_default = 50
        if self.data_source == 'nuplan':
            self.error_default = 5
        self.polygon_num_default = 16
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        polygon_num = kwargs['polygon_num'] if 'polygon_num' in kwargs else self.polygon_num_default
        save_path = self.save_path + f'_error_{max_error}'
        
        from shapely.geometry import Polygon, Point
        
        point_list = [self.traj[0]]
        index_list = [0]
        polygon = Polygon()
        for i in range(1, len(self.traj)):
            def get_polygon(point0, point1, index0, index1, r):
                c = (self.t_list[index0 + 1] - self.t_list[index0]) / (self.t_list[index1] - self.t_list[index0])
                point = point0 + (point1 - point0) * c
                point_list = []
                for j in range(polygon_num):
                    angle = (2 * j - 1) * np.pi / polygon_num
                    point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) * c)
                return Polygon(point_list)
            
            temp_polygon = get_polygon(point_list[-1], self.traj[i], index_list[-1], i, max_error)
            if polygon.is_empty:
                polygon = temp_polygon
            else:
                center_point = np.array([polygon.centroid.x, polygon.centroid.y])
                new_polygon = polygon.intersection(temp_polygon)
                if new_polygon.is_empty:
                    c = (self.t_list[index_list[-1] + 1] - self.t_list[index_list[-1]]) / (self.t_list[i - 1] - self.t_list[index_list[-1]])
                    add_point = self.traj[i - 1]
                    temp_point = point_list[-1] + (add_point - point_list[-1]) * c
                    if not polygon.contains(Point(temp_point)):
                        add_point = point_list[-1] + (center_point - point_list[-1]) / c
                    
                    point_list.append(add_point)
                    index_list.append(i - 1)
                    polygon = get_polygon(add_point, self.traj[i], index_list[-1], i, max_error)
                else:
                    polygon = new_polygon
        
        add_point = self.traj[-1]
        if not polygon.is_empty:
            center_point = np.array([polygon.centroid.x, polygon.centroid.y])
            c = (self.t_list[index_list[-1] + 1] - self.t_list[index_list[-1]]) / (self.t_list[len(self.traj) - 1] - self.t_list[index_list[-1]])
            temp_point = point_list[-1] + (add_point - point_list[-1]) * c
            if not polygon.contains(Point(temp_point)):
                add_point = point_list[-1] + (center_point - point_list[-1]) / c
        
        point_list.append(add_point)
        index_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                for i in range(0, len(index_list)):
                    write_bits.extend(double2bitarray(self.t_list[index_list[i]]))
                    for dim in range(traj_dim):
                        write_bits.extend(double2bitarray(point_list[i][dim]))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, **kwargs):
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        save_path = self.save_path + f'_error_{max_error}'
        traj_dim = self.traj.shape[1]
        
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            t_list = []
            point_list = []
            while read_index < len(read_bits) - 8:
                t = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                point = []
                for dim in range(traj_dim):
                    temp = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    point.append(temp)
                t_list.append(t)
                point_list.append(np.array(point))
        
        t_list = np.array(t_list)
        point_list = np.array(point_list)
        
        self.decompress_traj = []
        index = 0
        for i in range(len(self.t_list)):
            t = self.t_list[i]
            while index < len(t_list) - 1 and t_list[index + 1] - t < -1e-4:
                index = index + 1
            if index == len(t_list) - 1:
                temp_point = point_list[index]
            else:
                temp_point = point_list[index] + (point_list[index + 1] - point_list[index]) * (t - t_list[index]) / (t_list[index + 1] - t_list[index])
            
            self.decompress_traj.append(temp_point)
            
        self.decompress_traj = np.array(self.decompress_traj)

class SQUISHNOENCODECompressor(Compressor):
    def __init__(self, path, data_source='nuplan'):
        super().__init__('SQUISH-E-No-Encode', path, data_source)
        if self.data_source == 'geolife':
            self.error_default = 50
        if self.data_source == 'geolife_3d':
            self.error_default = 50
        if self.data_source == 'mopsi':
            self.error_default = 50
        if self.data_source == 'nuplan':
            self.error_default = 5
        pass
    
    @profile
    def compress(self, save=True, **kwargs):
        traj_dim = self.traj.shape[1]
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        compression_ratio = kwargs['compression_ratio'] if 'compression_ratio' in kwargs else 1
        save_path = self.save_path + f'_error_{max_error}'
        
        capacity = len(self.traj) / compression_ratio
        heap = []
        
        pi = np.zeros(len(self.traj))
        succ = np.zeros(len(self.traj), dtype=int) - 1
        pred = np.zeros(len(self.traj), dtype=int) - 1
        
        dict = {}
        
        for i in range(len(self.traj)):
            def adjust(j):
                if pred[j] != -1 and succ[j] != -1:
                    temp_point = self.traj[pred[j]] + (self.traj[succ[j]] - self.traj[pred[j]]) / (self.t_list[succ[j]] - self.t_list[pred[j]]) * (self.t_list[j] - self.t_list[pred[j]])
                    temp = pi[j] + np.linalg.norm(self.traj[j] - temp_point)
                    heapq.heappush(heap, (temp, j))
                    dict[j] = temp
            
            def reduce():
                data = heapq.heappop(heap)
                p, j = data[0], data[1]
                while j in dict and dict[j] != p:
                    data = heapq.heappop(heap)
                    p, j = data[0], data[1]
                
                pi[succ[j]] = max(p, pi[succ[j]])
                pi[pred[j]] = max(p, pi[pred[j]])
                succ[pred[j]] = succ[j]
                pred[succ[j]] = pred[j]
                adjust(pred[j])
                adjust(succ[j])
                
            
            heapq.heappush(heap, (np.inf, i))
            pi[i] = 0
            if i > 0:
                succ[i - 1] = i
                pred[i] = i - 1
                adjust(i - 1)
            if i + 1 > capacity:
                reduce()
                
        data = heapq.heappop(heap)
        p, j = data[0], data[1]
        while j in dict and dict[j] != p:
            data = heapq.heappop(heap)
            p, j = data[0], data[1]
        while p < max_error:
            pi[succ[j]] = max(p, pi[succ[j]])
            pi[pred[j]] = max(p, pi[pred[j]])
            succ[pred[j]] = succ[j]
            pred[succ[j]] = pred[j]
            adjust(pred[j])
            adjust(succ[j])
            
            data = heapq.heappop(heap)
            p, j = data[0], data[1]
            while j in dict and dict[j] != p:
                data = heapq.heappop(heap)
                p, j = data[0], data[1]
    
        index_list = [0]
        while index_list[-1] != len(self.traj) - 1:
            index_list.append(succ[index_list[-1]])
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                for i in range(0, len(index_list)):
                    write_bits.extend(double2bitarray(self.t_list[index_list[i]]))
                    for dim in range(traj_dim):
                        write_bits.extend(double2bitarray(self.traj[index_list[i]][dim]))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, **kwargs):
        max_error = kwargs['max_error'] if kwargs['max_error'] != None else self.error_default
        save_path = self.save_path + f'_error_{max_error}'
        traj_dim = self.traj.shape[1]
        
        with open(save_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            t_list = []
            point_list = []
            while read_index < len(read_bits) - 8:
                t = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                point = []
                for dim in range(traj_dim):
                    temp = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    point.append(temp)
                t_list.append(t)
                point_list.append(np.array(point))
        
        t_list = np.array(t_list)
        point_list = np.array(point_list)
        
        self.decompress_traj = []
        index = 0
        for i in range(len(self.t_list)):
            t = self.t_list[i]
            while index < len(t_list) - 1 and t_list[index + 1] - t < -1e-4:
                index = index + 1
            if index == len(t_list) - 1:
                temp_point = point_list[index]
            else:
                temp_point = point_list[index] + (point_list[index + 1] - point_list[index]) * (t - t_list[index]) / (t_list[index + 1] - t_list[index])
            
            self.decompress_traj.append(temp_point)
            
        self.decompress_traj = np.array(self.decompress_traj)

import multiprocessing

def process_file(file_path,
                 data_source,
                 max_error,
                 utf,
                 accuracy,
                 block_size,
                 scale_length,
                 epsilon,):
    logs = f'{file_path}\n'
    
    result = []
    
    # squish_no_encode_compressor = SQUISHNOENCODECompressor(file_path, data_source)
    # # squish_no_encode_compressor.compress(max_error=max_error)
    # squish_no_encode_compressor.decompress(max_error=max_error)
    # squish_no_encode_result = squish_no_encode_compressor.evaluation_metrics(max_error)
    # # squish_no_encode_compressor.plot_traj()
    # logs += f"squish no encode performance: {squish_no_encode_result}\n"
    # result.append([squish_no_encode_compressor.name, squish_no_encode_result])
    
    # ciseds_no_encode_compressor = CISEDSNOENCODECompressor(file_path, data_source)
    # # ciseds_no_encode_compressor.compress(max_error=max_error)
    # ciseds_no_encode_compressor.decompress(max_error=max_error)
    # ciseds_no_encode_result = ciseds_no_encode_compressor.evaluation_metrics(max_error)
    # # ciseds_no_encode_compressor.plot_traj()
    # logs += f"cised s no encode performance: {ciseds_no_encode_result}\n"
    # result.append([ciseds_no_encode_compressor.name, ciseds_no_encode_result])
    
    # cisedw_no_encode_compressor = CISEDWNOENCODECompressor(file_path, data_source)
    # # cisedw_no_encode_compressor.compress(max_error=max_error)
    # cisedw_no_encode_compressor.decompress(max_error=max_error)
    # cisedw_no_encode_result = cisedw_no_encode_compressor.evaluation_metrics(max_error)
    # # cisedw_no_encode_compressor.plot_traj()
    # logs += f"cised w no encode performance: {cisedw_no_encode_result}\n"
    # result.append([cisedw_no_encode_compressor.name, cisedw_no_encode_result])
    
    # pilotc_no_enhanced_compressor = PILOTCNOENHANCEDZIGZAGCompressor(file_path, data_source)
    # # pilotc_no_enhanced_compressor.compress(max_error=max_error, utf=utf, accuracy=accuracy, block_size=block_size, scale_length=scale_length)
    # pilotc_no_enhanced_compressor.decompress(max_error=max_error, utf=utf, accuracy=accuracy, block_size=block_size, scale_length=scale_length)
    # pilotc_no_enhanced_result = pilotc_no_enhanced_compressor.evaluation_metrics(max_error)
    # # pilotc_no_enhanced_compressor.plot_traj()
    # logs += f"cised w no encode performance: {pilotc_no_enhanced_result}\n"
    # result.append([pilotc_no_enhanced_compressor.name, pilotc_no_enhanced_result])
    
    squish_compressor = SQUISHCompressor(file_path, data_source)
    # squish_compressor.compress(max_error=max_error, utf=utf, accuracy=accuracy)
    squish_compressor.decompress(max_error=max_error, utf=utf, accuracy=accuracy)
    squish_result = squish_compressor.evaluation_metrics(max_error)
    # squish_compressor.plot_traj()
    logs += f"squish performance: {squish_result}\n"
    result.append([squish_compressor.name, squish_result])
    
    ciseds_compressor = CISEDSCompressor(file_path, data_source)
    # ciseds_compressor.compress(max_error=max_error, utf=utf, accuracy=accuracy)
    ciseds_compressor.decompress(max_error=max_error, utf=utf, accuracy=accuracy)
    ciseds_result = ciseds_compressor.evaluation_metrics(max_error)
    # ciseds_compressor.plot_traj()
    logs += f"cised s performance: {ciseds_result}\n"
    result.append([ciseds_compressor.name, ciseds_result])
    
    cisedw_compressor = CISEDWCompressor(file_path, data_source)
    # cisedw_compressor.compress(max_error=max_error, utf=utf, accuracy=accuracy)
    cisedw_compressor.decompress(max_error=max_error, utf=utf, accuracy=accuracy)
    cisedw_result = cisedw_compressor.evaluation_metrics(max_error)
    # cisedw_compressor.plot_traj()
    logs += f"cised w performance: {cisedw_result}\n"
    result.append([cisedw_compressor.name, cisedw_result])
    
    pilotc_compressor = PILOTCCompressor(file_path, data_source)
    # pilotc_compressor.compress(max_error=max_error, utf=utf, accuracy=accuracy, block_size=block_size, scale_length=scale_length, epsilon=epsilon)
    pilotc_compressor.decompress(max_error=max_error, utf=utf, accuracy=accuracy, block_size=block_size, scale_length=scale_length, epsilon=epsilon)
    pilotc_result = pilotc_compressor.evaluation_metrics(max_error)
    # pilotc_compressor.plot_traj()
    logs += f"pilot c performance: {pilotc_result}\n"
    result.append([pilotc_compressor.name, pilotc_result])
    
    logs += '\n'
    return result, logs

def main():
    data_sources = ['nuplan', 'geolife', 'mopsi']
    # data_sources = ['geolife_3d']
    
    title = {'nuplan': '(1) nuPlan', 'geolife': '(2) GeoLife', 'mopsi': '(3) Mopsi', 'geolife_3d': 'GeoLife-3D'}
    
    color = {'PILOT-C': 'r', 'CISED-S': 'g', 'CISED-W': 'b', 'SQUISH-E': 'y', 'PILOT-C-No-Enhanced-Zigzag': 'r', 'CISED-S-No-Encode': 'g', 'CISED-W-No-Encode': 'b', 'SQUISH-E-No-Encode': 'y',
             'nuplan': 'r', 'geolife':'g', 'mopsi': 'b'}
    linestyle = {'PILOT-C': '-', 'CISED-S': '-', 'CISED-W': '-', 'SQUISH-E': '-', 'PILOT-C-No-Enhanced-Zigzag': '--', 'CISED-S-No-Encode': '--', 'CISED-W-No-Encode': '--', 'SQUISH-E-No-Encode': '--',
                 'nuplan': '-', 'geolife':'-', 'mopsi': '-'}
    markerstyle = {'PILOT-C': '.', 'CISED-S': 'o', 'CISED-W': 's', 'SQUISH-E': 'x', 'PILOT-C-No-Enhanced-Zigzag': '.', 'CISED-S-No-Encode': 'o', 'CISED-W-No-Encode': 's', 'SQUISH-E-No-Encode': 'x',
                   'nuplan': '.', 'geolife':'o', 'mopsi': 's'}
    
    plt.rcParams['font.size'] = 16
    fig, axs = plt.subplots(1, len(data_sources))
    # fig, axs = plt.subplots(1, 1)
    fig.set_figwidth(20)
    # fig.set_figwidth(7)
    fig.set_figheight(4)
    # fig.set_figheight(4.5)
    # fig.set_figheight(5)
    
    for i in range(len(data_sources)):
        data_source = data_sources[i]
        if data_source == 'geolife':
            path = '/home/wkf/data/PRESS/datasets/geolife'
        if data_source == 'geolife_3d':
            path = '/home/wkf/data/PRESS/datasets/geolife_3d'
        if data_source == 'nuplan':
            path = '/nas/common/data/trajectory/nuplan/nuplan_csv/test'
        if data_source == 'shangqi':
            path = '/nas/common/data/trajectory/ShangQi-2023/saic_data/long_term/eval_2023_07_24_10_53_17_'
        if data_source == 'mopsi':
            path = '/home/wkf/data/PRESS/datasets/mopsi'
        
        del_csvs = []
        
        tot_path = []
        
        if data_source == 'geolife':
            for file_name in os.listdir(path):
                ff = os.path.join(path, file_name)
                tot_path.append(ff)
        if data_source == 'geolife_3d':
            for file_name in os.listdir(path):
                ff = os.path.join(path, file_name)
                tot_path.append(ff)
        if data_source == 'mopsi':
            for file_name in os.listdir(path):
                ff = os.path.join(path, file_name)
                tot_path.append(ff)
        if data_source == 'nuplan':
            for dir_name in os.listdir(path):
                ff = os.path.join(path, dir_name)
                if ff in del_csvs:
                    continue
                tot_path.append(ff)
        if data_source == 'shangqi':
            tot_path.append(path)
        
        multiprocessing_flag = True
        
        test_utf = [None]
        # test_utf = [2, 3, 4, 5, 6, 7, 8]
        
        test_error = [None]
        test_error = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if data_source != 'nuplan' else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        test_accuracy = [None]
        # test_accuracy = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        
        test_scale = [None]
        # test_scale = [2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0]
        test_scale_x = [-7, -6, -5, -4, -3, -2, -1, 0]
        test_scale_labels = [r'$2^{-7}$', r'$2^{-6}$', r'$2^{-5}$', r'$2^{-4}$', r'$2^{-3}$', r'$2^{-2}$', r'$2^{-1}$', r'$2^{0}$']
        
        test_block_size = [None]
        # test_block_size = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
        
        test_epsilon = [None]
        # test_epsilon = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        
        data = {}
        data2 = {}
        
        temp = [0, 0, 0, 0]
        
        combinations = itertools.product(test_utf, test_error, test_accuracy, test_scale, test_block_size, test_epsilon)
        
        for combination in combinations:
            utf, error, accuracy, scale_length, block_size, epsilon = combination
            global_counter = Global_Counter()
            results = []
            logs = ''
            
            if multiprocessing_flag:
                with multiprocessing.Pool(processes=16) as pool:
                    pool_results = []
                    for ff in tot_path:
                        result = pool.apply_async(process_file, (ff,
                                                                data_source,
                                                                error,
                                                                utf,
                                                                accuracy,
                                                                block_size,
                                                                scale_length,
                                                                epsilon,))
                        pool_results.append(result)
                    
                    for result in pool_results:
                        res, log = result.get()
                        results.extend(res)
                        logs += log
            else:
                for ff in tot_path:
                    result, log = process_file(ff,
                                            data_source,
                                            error,
                                            utf,
                                            accuracy,
                                            block_size,
                                            scale_length,
                                            epsilon,)
                    results.extend(result)
                    logs += log
                
            for r in results:
                global_counter.count(r[0], r[1])
            
            for key in global_counter.counter:
                if key not in data:
                    data[key] = []
                    data2[key] = []
                # data[key].append(1 / global_counter.counter[key]['compress_ratio'] * 100)
                # data[key].append(np.log10(1 / global_counter.counter[key]['compress_ratio'] * 100))
                data[key].append(global_counter.counter[key]['mean_error'])
                # data[key].append(global_counter.counter[key]['total_error_cnt'] / global_counter.counter[key]['total_length'] * 100)
                # data2[key].append((3 if data_source =='nuplan' else 2) + np.log10(global_counter.counter[key]['total_error_cnt'] / global_counter.counter[key]['total_length'] * 100))
                # data2[key].append(3 + np.log10(global_counter.counter[key]['total_error_cnt'] / global_counter.counter[key]['total_length'] * 100))
            
            
            # temp[0] += global_counter.counter['SQUISH-E']['mean_error'] / global_counter.counter['PILOT-C']['mean_error']
            # temp[1] += global_counter.counter['CISED-S']['mean_error'] / global_counter.counter['PILOT-C']['mean_error']
            # temp[2] += global_counter.counter['CISED-W']['mean_error'] / global_counter.counter['PILOT-C']['mean_error']
            # temp[3] += global_counter.counter['PILOT-C']['mean_error'] / global_counter.counter['PILOT-C-No-Enhanced-Zigzag']['mean_error']
            
            print(combination)
            for key in global_counter.counter:
                print(key, global_counter.counter[key]['var_error'])
            
            if multiprocessing_flag:
                with open(f'logs/{data_source}_error_{error}_utf_{utf}_accuracy_{accuracy}.txt', 'w') as f:
                    f.write(logs)
                    f.write(global_counter.get_result())
            else:
                print(logs)
                print(global_counter.get_result())
        
        # for key in data:
        #     axs.plot(np.array(test_block_size), data[key], label=data_source, color=color[data_source], linestyle=linestyle[data_source], marker=markerstyle[data_source], markerfacecolor='none')
        
        # print(data_source, temp)
        
        for key in data:
            axs[i].plot(np.array(test_error), data[key], label=key, color=color[key], linestyle=linestyle[key], marker=markerstyle[key], markerfacecolor='none')
            # axs[i].plot(np.array(test_epsilon), data[key], label='Compression ratio', color=color[key], linestyle=linestyle[key], marker=markerstyle[key], markerfacecolor='none')
        axs[i].set_xlabel(r'$\epsilon$ (meters)')
        # axs[i].set_xlabel(r'Length of the chunks (bits)')
        # axs[i].set_xlabel(r'$\epsilon_p$ ($\times \epsilon$ meters)')
        # axs[i].set_xlabel(r'$\epsilon_f$ ($\times \epsilon$ meters)')
        # axs[i].set_xlabel(r'$r_{ret}$')
        # axs[i].set_ylabel('Compression ratio (%)')
        axs[i].set_ylabel('Average error (meters)')
        axs[i].set_title(title[data_source])
        # if data_source == 'nuplan':
        #     axs[i].set_yticks([0.10, 0.11, 0.12, 0.13])
        
        # axs[i].set_ylim(bottom = 0)
        # if data_source != 'nuplan':
        #     axs[i].set_yticks([-0.5, 0, 0.5, 1], [r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$', r'$10^{1}$'])
        # else:
        #     axs[i].set_yticks([-1, -0.5, 0, 0.5], [r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$', r'$10^{0.5}$'])
        lines, labels = axs[i].get_legend_handles_labels()
        # axs[i].set_xticks(test_scale_x, test_scale_labels)
        
        # axs[i].tick_params(axis='y')
        # ax2 = axs[i].twinx()
        # ax2.set_ylabel('Percentage of error points (%)')
        # for key in data2:
        #     ax2.bar(test_epsilon, data2[key], color='b', alpha=0.6, width=0.05, label='Percentage of error points')
        # ax2.tick_params(axis='y')
        # if data_source != 'nuplan':
        #     ax2.set_yticks([0, 1, 2, 3], [r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
        # else:
        #     ax2.set_yticks([0, 1, 2, 3], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        # # ax2.set_yticks([0, 1, 2, 3, 4], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
        
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # lines.extend(lines2)
        # labels.extend(labels2)
    
    # axs.legend(loc='upper left', fontsize=12)
    # axs.set_xlabel(r'$b_s$')
    # axs.set_ylabel('Compression ratio (%)')
    
    plt.tight_layout()
    # fig.legend(lines, labels, loc = 'upper center', ncol=4, bbox_to_anchor=(0.5, 1))
    plt.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.17)
    # plt.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.15)
    # plt.subplots_adjust(left=0.06, right=0.99, top=0.75, bottom=0.13)
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.17)
    # plt.subplots_adjust(left=0.06, right=0.95, top=0.8, bottom=0.15)
    # plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.17)
    # plt.savefig(f'images/test_average_error.pdf')

if __name__ == '__main__':
    main()