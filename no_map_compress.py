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

np.random.seed(0)

if 'LINE_PROFILE_IT' not in os.environ:
    def profile(x):
        return x
    
@profile
def get_dp(compressed_result, length_bit, exp_bit):
    max_bits = 0
    
    need_bits = []
    for i in range(len(compressed_result)):
        if compressed_result[i] == 0:
            need_bits.append(0)
            continue
        temp_bits = 1
        while compressed_result[i] > 2 ** (temp_bits - 1) - 1 or compressed_result[i] < -2 ** (temp_bits - 1):
            temp_bits = temp_bits + 1 
        need_bits.append(temp_bits)
        max_bits = max(max_bits, temp_bits)
    max_bits = max_bits + 1
    assert max_bits <= 32, "max_bits should not be greater than 32"
    
    dp = np.full((len(compressed_result) + 1, max_bits), np.inf).tolist()
    last_index = np.zeros((len(compressed_result) + 1, max_bits)).tolist()
    
    for j in range(max_bits):
        dp[0][j] = exp_bit + length_bit
    
    for i in range(len(compressed_result)):
        argmin_dp_i = np.argmin(dp[i])
        for j in range(need_bits[i], max_bits):
            if dp[i][argmin_dp_i] + exp_bit + length_bit + j > dp[i][j] + j:
                dp[i + 1][j] = dp[i][j] + j
                last_index[i + 1][j] = j
            else:
                dp[i + 1][j] = dp[i][argmin_dp_i] + exp_bit + length_bit + j
                last_index[i + 1][j] = argmin_dp_i
    for j in range(max_bits):
        dp[0][j] = 0
    
    return dp, last_index

def get_compressed_result(compressed_result, dp, last_index, end_index):
    argmin_dp = int(np.argmin(dp[end_index]))
    temp_result = []
    for i in range(end_index, 0, -1):
        temp_result.append(argmin_dp)
        argmin_dp = int(last_index[i][argmin_dp])
    temp_result.reverse()

    final_result = []
    
    index = 0
    while index < end_index:
        temp_list = [compressed_result[index]]
        while index + 1 < end_index and temp_result[index] == temp_result[index + 1]:
            index = index + 1
            temp_list.append(compressed_result[index])
        final_result.append([temp_result[index], temp_list])
        index = index + 1
    return final_result

def lossless_compress(compressed_result, length_bit, exp_bit, bits = None):
    dp, last_index = get_dp(compressed_result, length_bit, exp_bit)
    
    if bits is None:
        end_index = len(compressed_result)
    else:
        end_index = 0
        for i in range(len(compressed_result), 0, -1):
            if np.min(dp[i]) <= bits:
                end_index = i
                break
    
    final_result = get_compressed_result(compressed_result, dp, last_index, end_index)
    return final_result, min(dp[end_index])

class NoMapCompressor():
    '''
    base class
    '''
    _cache_version = 'v20250313'
    _cache_attributes = {
        'traj',
        't_list',
    }
    
    def __init__(self, name, path, data_source='nuplan', time_unit=0.1, drop=0):
        path_name = path.split("/")[-1]
        self.name = name
        self.data_source = data_source
        self.path = path
        self.time_unit = time_unit
        self.drop = drop
        self.cache_path = f'/home/wkf/data/PRESS/cache/{data_source}/drop_{drop}/{path_name}/'
        self.result_path = f'/home/wkf/data/PRESS/result/{data_source}/drop_{drop}/{path_name}/'
        self.save_path = self.result_path + f'compressed_{name}'
        self.load_path = self.result_path + f'compressed_{name}'
        self.origin_path = self.result_path + f'origin'
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        if data_source == 'nuplan':
            self.load_nuplan_traj()
        elif data_source == 'geolife':
            self.load_geolife_traj()
        elif data_source == 'shangqi':
            self.load_shangqi_traj()
        elif data_source == 'mopsi':
            self.load_mopsi_traj()
        elif data_source == 'uav':
            self.load_uav_traj()
        elif data_source == 'geolife_3d':
            self.load_geolife_3d_traj()
        
        self.decompress_traj = None
        
    def compress(self, save_path=None, save=True, **kwargs):
        pass

    def decompress(self, load_path=None, **kwargs):
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
            
            traj = np.array(traj)
            t_list = np.array(t_list)
            length = len(traj)
            new_length = int(length * (1 - self.drop))
            
            use_drop = np.arange(1, length - 1)
            np.random.shuffle(use_drop)
            use_drop = use_drop[:(new_length - 2)]
            use_drop = np.append(use_drop, 0)
            use_drop = np.append(use_drop, length - 1)
            use_drop = np.sort(use_drop)
            
            self.traj = traj[use_drop]
            self.t_list = t_list[use_drop]
            
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
            new_length = int(length * (1 - self.drop))
            
            use_drop = np.arange(1, length - 1)
            np.random.shuffle(use_drop)
            use_drop = use_drop[:(new_length - 2)]
            use_drop = np.append(use_drop, 0)
            use_drop = np.append(use_drop, length - 1)
            use_drop = np.sort(use_drop)
            
            self.traj = traj[use_drop]
            self.t_list = t_list[use_drop]
            
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
        error = np.sqrt(np.sum((self.traj - self.decompress_traj) ** 2, axis=1))
        
        error_cnt = np.where(error > error_bound)[0]
        
        max_error = np.max(error)
        mean_error = np.mean(error)
        self.argmax_error = int(np.argmax(error))
        return {'max_error': float(max_error),
                'mean_error': float(mean_error),
                'compress_ratio': float((len(self.traj) * (len(self.traj[0]) + 1) * 8) / self.compress_bits * 8),
                'length': len(self.traj),
                'error_cnt': len(error_cnt)}
        
    def plot_traj(self, argmax_error=None):
        if self.decompress_traj is None:
            print("Please decompress first")
            return
        
        self.decompress_traj = np.array(self.decompress_traj)
        if self.traj.shape[1] == 2:
            if argmax_error is None:
                plt.scatter(self.traj[:, 0], self.traj[:, 1], label='origin', s=0.01)
                plt.scatter(self.decompress_traj[:, 0], self.decompress_traj[:, 1], label='decompress', s=0.01)
            else:
                start_index = argmax_error - 60
                end_index = argmax_error + 60
                
                plt.scatter(self.traj[start_index: end_index, 0], self.traj[start_index: end_index, 1], label='origin', s=0.1)
                plt.scatter(self.decompress_traj[start_index: end_index, 0], self.decompress_traj[start_index: end_index, 1], label='decompress', s=0.1)
            plt.axis('equal')    
            plt.legend()
            plt.savefig(f'images/test_{self.name}.png', dpi=1000)
            plt.gcf().clear()
        elif self.traj.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.traj[:, 0], self.traj[:, 1], self.traj[:, 2], label='origin', s=0.1)
            ax.scatter(self.decompress_traj[:, 0], self.decompress_traj[:, 1], self.decompress_traj[:, 2], label='decompress', s=0.1)
            ax.legend()
            ax.axis('equal')
            plt.savefig(f'images/test_{self.name}.png', dpi=1000)
            plt.gcf().clear()
                    
        

class DCTNoMapCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', time_unit=0.1, drop=0):
        super().__init__('dct_no_map', path, data_source, time_unit, drop)
        pass
    
    @profile
    def compress(self, save_path=None, save=True, **kwargs):
        if save_path is None:
            save_path = self.save_path
        
        block_size = kwargs['block_size'] if 'block_size' in kwargs else 300
        compression_ratio = kwargs['compression_ratio'] if 'compression_ratio' in kwargs else None
        dynamic_ratio_flag = kwargs['dynamic_ratio_flag'] if 'dynamic_ratio_flag' in kwargs else False
        scale = kwargs['scale'] if 'scale' in kwargs else 1
        scale_length = kwargs['scale_length'] if 'scale_length' in kwargs else 1
        max_error = kwargs['max_error'] if'max_error' in kwargs else None
        
        self.scale_length = scale_length
        
        traj_dim = self.traj.shape[1]
        max_bits = (traj_dim + 1) * 64 * len(self.traj) / compression_ratio // 8 * 8
        
        other_points = []
        index_list = []
        
        start_index = 0
        end_index = 0
        while start_index < len(self.t_list):
            end_index = start_index
            while end_index < len(self.t_list) - 1 and \
                  self.t_list[end_index + 1] - self.t_list[end_index] < 120 and \
                  np.sqrt(np.sum((self.traj[end_index + 1] - self.traj[end_index]) ** 2) / (self.t_list[end_index + 1] - self.t_list[end_index])) < 1000:
                end_index = end_index + 1
            
            if end_index - start_index <= 0:
                for i in range(start_index, end_index + 1):
                    other_points.append(i)
                start_index = end_index + 1
                continue
            index_list.append([start_index, end_index])
            start_index = end_index + 1
         
        result_list = []
        tot_bits = 32 + (traj_dim + 1) * 64 * len(other_points) + 192 * (len(other_points) != len(self.traj))
        max_bits = (traj_dim + 1) * 64 * (len(self.traj) - len(other_points)) / compression_ratio // 8 * 8 + (traj_dim + 1) * 64 * len(other_points)
        
        
        if dynamic_ratio_flag:
            all_compress_data = []
            all_real_data = []
            all_dp = []
            all_last_index = []
            all_length = []
            all_mean = []
            all_block_size = []
            heap = []
            index_dict = {}
            for i in range(len(index_list)):
                start_index, end_index = index_list[i]
                temp_compress_data = [[] for _ in range(traj_dim)]
                temp_real_data = [[] for _ in range(traj_dim)]
                temp_dp = [[] for _ in range(traj_dim)]
                temp_last_index = [[] for _ in range(traj_dim)]
                temp_mean = [[] for _ in range(traj_dim)]
                def cal_delta_old(start_index, end_index):
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
                    delta = temp_traj[1:] - temp_traj[:-1]
                    return delta, temp_traj
                
                delta, temp_traj = cal_delta_old(start_index, end_index)
                
                length = end_index - start_index + 1
                
                temp_block_num = (len(delta) - 1) // block_size + 1
                temp_block_size = (len(delta) - 1) // temp_block_num + 1
                all_length.append(len(delta))
                all_block_size.append(temp_block_size)
                tot_bits = tot_bits + 128 + traj_dim * 64
                
                # for k in range(0, len(temp_traj), temp_block_size):
                #     index = k // temp_block_size
                #     xi, yi = temp_traj[k]
                #     plt.annotate(f'{index}', (xi, yi), fontsize=3)
                # plt.scatter(temp_traj[::temp_block_size, 0], temp_traj[::temp_block_size, 1], label='divide', s=1)
                # plt.scatter(temp_traj[:, 0], temp_traj[:, 1], label='origin', s=0.1)
                # plt.axis('equal')    
                # plt.legend()
                # plt.savefig(f'images/test_{i}_block{temp_block_size}.png', dpi=1000)
                # plt.gcf().clear()
                
                length_bit = max(0, math.ceil(math.log2(temp_block_size * scale_length)))
                
                for dim in range(traj_dim):
                    for block_id in range(temp_block_num):
                        block_data = delta[block_id * temp_block_size: min((block_id + 1) * temp_block_size, len(delta)), dim]
                        mean = np.mean(block_data)
                        block_data = block_data - mean
                        
                        dctn_data = dctn(block_data)
                        compressed_data = []
                        index = int((len(dctn_data) - 1) * scale_length)
                        while int(np.round(dctn_data[index] * scale)) == 0 and index > 0:
                            index = index - 1
                        
                        compressed_data = np.round(dctn_data[1: index + 1] * scale).astype(int).tolist()
                        
                        dp, last_index = get_dp(compressed_data, length_bit, 5)
                        temp_compress_data[dim].append(compressed_data)
                        temp_real_data[dim].append(dctn_data)
                        temp_dp[dim].append(dp)
                        temp_last_index[dim].append(last_index)
                        temp_mean[dim].append(mean)
                        tot_bits = tot_bits + min(dp[index]) + length_bit + 32
                        if index > 0:
                            heapq.heappush(heap, (-index / temp_block_size, i, dim, block_id, index))
                        else:
                            index_dict[(i, dim, block_id)] = 0
                
                all_dp.append(temp_dp)
                all_last_index.append(temp_last_index)
                all_compress_data.append(temp_compress_data)
                all_real_data.append(temp_real_data)
                all_mean.append(temp_mean)
            
            while len(heap) > 0 and tot_bits > max_bits:
                heap_top = heapq.heappop(heap)
                ratio = heap_top[0]
                i = heap_top[1]
                dim = heap_top[2]
                block_id = heap_top[3]
                index = heap_top[4]
                
                tot_bits = tot_bits - min(all_dp[i][dim][block_id][index]) + min(all_dp[i][dim][block_id][index - 1])
                if index - 1 > 0:
                    heapq.heappush(heap, (-(index - 1) / all_block_size[i], i, dim, block_id, index - 1))
                else:
                    index_dict[(i, dim, block_id)] = 0
                    
            while heap:
                heap_top = heapq.heappop(heap)
                i = heap_top[1]
                dim = heap_top[2]
                block_id = heap_top[3]
                index = heap_top[4]
                
                index_dict[(i, dim, block_id)] = index
                
            for i in range(len(index_list)):
                start_index, end_index = index_list[i]
                temp_dp = all_dp[i]
                temp_last_index = all_last_index[i]
                temp_compress_data = all_compress_data[i]
                temp_mean = all_mean[i]
                
                result = [[] for _ in range(traj_dim)]
                
                for dim in range(traj_dim):
                    for block_id in range(len(temp_dp[dim])):
                        dp = temp_dp[dim][block_id]
                        last_index = temp_last_index[dim][block_id]
                        compressed_dim = temp_compress_data[dim][block_id]
                        index = index_dict[(i, dim, block_id)]
                        compressed_result = get_compressed_result(compressed_dim, dp, last_index, index)
                        result[dim].append([compressed_result, temp_mean[dim][block_id]])
                
                result_list.append([start_index, all_block_size[i], all_length[i], result])
            
            self.compress_bits = tot_bits
        else:
            for i in range(len(index_list)):
                start_index, end_index = index_list[i]
                def cal_delta_old(start_index, end_index):
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
                    delta = temp_traj[1:] - temp_traj[:-1]
                    return delta
                
                delta = cal_delta_old(start_index, end_index)
                
                length = end_index - start_index + 1
                result = [[] for _ in range(traj_dim)]
                
                temp_block_num = (len(delta) - 1) // block_size + 1
                temp_block_size = (len(delta) - 1) // temp_block_num + 1
                
                length_bit = max(0, math.ceil(math.log2(temp_block_size * scale_length)))
                
                temp_max_bits = ((traj_dim + 1) * 64 * length / compression_ratio - 160 / len(self.traj) * length) - (128 + 64 * traj_dim)
                tot_bits = tot_bits + (128 + 64 * traj_dim)
                
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
                        
                        dp, last_index = get_dp(compressed_dim, length_bit, 5)
                        lossless_compressed_dim, bit_cnt = lossless_compress(compressed_dim, length_bit, 5, temp_max_bits * len(block_dim) / len(delta) / 2 - 40)
                        tot_bits = tot_bits + bit_cnt + length_bit + 32
                        result[dim].append([lossless_compressed_dim, mean])
                
                result_list.append([start_index, temp_block_size, len(delta), result])
        
        if max_error is not None:
            decompress_traj0 = []
            t_list0 = []
            for i in range(len(result_list)):
                start_index = result_list[i][0]
                start_t = self.t_list[start_index]
                start_point = self.traj[start_index]
                block_size = result_list[i][1]
                delta_num = result_list[i][2]
                result = result_list[i][3]
                
                tot_idct = np.array([[] for _ in range(traj_dim)])
                for dim in range(self.traj.shape[1]):
                    for i in range(len(result[dim])):
                        compressed_dim = []
                        for _, data in result[dim][i][0]:
                            compressed_dim.extend(data)
                        mean = np.float32(result[dim][i][1])
                        compressed_dim = np.pad(compressed_dim, (1, min(block_size, delta_num - block_size * i) - len(compressed_dim) - 1), 'constant')
                        idct_dim = idctn(compressed_dim / scale)
                        idct_dim = idct_dim + mean
                        tot_idct[dim] = np.concatenate((tot_idct[dim], idct_dim))
                
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
                while index0 < len(t_list0) - 1 and t_list0[index0 + 1] - t < -1e-4:
                    index0 = index0 + 1
                while index1 < len(t_list1) and t_list1[index1] - t < -1e-4:
                    index1 = index1 + 1
                    
                if index0 == len(t_list0) - 1:
                    temp_point = decompress_traj0[index0]
                else:
                    temp_point = decompress_traj0[index0] + (decompress_traj0[index0 + 1] - decompress_traj0[index0]) * (t - t_list0[index0]) / (t_list0[index0 + 1] - t_list0[index0])
                
                if index1 != len(t_list1) and abs(t_list1[index1] - t) < 1e-4:
                    all_decompress_traj.append(decompress_traj1[index1])
                else:
                    all_decompress_traj.append(temp_point)
            all_decompress_traj = np.array(all_decompress_traj)
            
            error = np.sqrt(np.sum((self.traj - all_decompress_traj) ** 2, axis=1))
            error_cnt = np.where(error > max_error)[0]
            for i in error_cnt:
                other_points.append(i)
            
            other_points.sort()
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(signint2bitarray(len(result_list), 32))
                if len(result_list) != 0:
                    write_bits.extend(double2bitarray(self.time_unit))
                    write_bits.extend(double2bitarray(scale))
                    write_bits.extend(double2bitarray(scale_length))
                
                for i in range(len(result_list)):
                    start_index = result_list[i][0]
                    temp_block_size = result_list[i][1]
                    delta_num = result_list[i][2]
                    result = result_list[i][3]
                    
                    length_bit = max(0, math.ceil(math.log2(temp_block_size * scale_length)))
                
                    write_bits.extend(double2bitarray(self.t_list[start_index]))
                    write_bits.extend(signint2bitarray(temp_block_size, 32))
                    write_bits.extend(signint2bitarray(delta_num, 32))
                    
                    for dim in range(traj_dim):
                        write_bits.extend(double2bitarray(self.traj[start_index][dim]))
                        for i in range(len(result[dim])):
                            write_bits.extend(signint2bitarray(len(result[dim][i][0]), length_bit))
                            for j in range(len(result[dim][i][0])):
                                write_bits.extend(signint2bitarray(result[dim][i][0][j][0], 5))
                                write_bits.extend(signint2bitarray(len(result[dim][i][0][j][1]), length_bit))
                                for k in range(len(result[dim][i][0][j][1])):
                                    write_bits.extend(signint2bitarray(result[dim][i][0][j][1][k], result[dim][i][0][j][0]))
                            write_bits.extend(float2bitarray(result[dim][i][1]))
                        
                for i in range(len(other_points)):
                    write_bits.extend(double2bitarray(self.t_list[other_points[i]]))
                    for dim in range(traj_dim):
                        write_bits.extend(double2bitarray(self.traj[other_points[i]][dim]))
                
                self.compress_bits = len(write_bits)
                assert self.compress_bits == tot_bits, (self.compress_bits, tot_bits)
                write_bits.tofile(f)
    
    @profile
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        result_list = []
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            self.compress_bits = len(read_bits)
            read_index = 0
            
            num = bitarray2signint(read_bits[read_index:read_index+32], False)
            read_index += 32
            other_points = []
            traj_dim = self.traj.shape[1]
            
            if num != 0:
                time_unit = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                scale = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                scale_length = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
            
            for i in range(num):
                start_t = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                block_size = bitarray2signint(read_bits[read_index:read_index+32], False)
                read_index += 32
                delta_num = bitarray2signint(read_bits[read_index:read_index+32], False)
                read_index += 32
                
                result_num = (delta_num - 1) // block_size + 1
                length_bit = max(0, math.ceil(math.log2(block_size * scale_length)))
                
                start_point = []
                result = []
                
                for dim in range(traj_dim):
                    start_dim = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    result_dim = []
                    for _ in range(result_num):
                        block_dim_num = bitarray2signint(read_bits[read_index:read_index+length_bit], False)
                        read_index += length_bit
                        compressed_dim = []
                        for j in range(block_dim_num):
                            bits = bitarray2signint(read_bits[read_index:read_index+5], False)
                            read_index += 5
                            temp_dim_num = bitarray2signint(read_bits[read_index:read_index+length_bit], False)
                            read_index += length_bit
                            for k in range(temp_dim_num):
                                compressed_dim.append(bitarray2signint(read_bits[read_index:read_index+bits]))
                                read_index += bits
                        mean = bitarray2float(read_bits[read_index:read_index+32])
                        read_index += 32
                        result_dim.append([compressed_dim, mean])
                    start_point.append(start_dim)
                    result.append(result_dim)
                
                result_list.append([start_t, start_point, block_size, delta_num, result])
                
            while read_index < len(read_bits) - 8:
                t = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                point = []
                
                for i in range(traj_dim):
                    point.append(bitarray2double(read_bits[read_index:read_index+64]))
                    read_index += 64
                other_points.append([t, point])
        
        decompress_traj0 = []
        t_list0 = []
        for i in range(len(result_list)):
            start_t = result_list[i][0]
            start_point = result_list[i][1]
            block_size = result_list[i][2]
            delta_num = result_list[i][3]
            result = result_list[i][4]
            
            tot_idct = [[] for _ in range(traj_dim)]
            for dim in range(self.traj.shape[1]):
                for i in range(len(result[dim])):
                    compressed_dim = result[dim][i][0]
                    mean = np.float32(result[dim][i][1])
                    compressed_dim = np.pad(compressed_dim, (1, min(block_size, delta_num - block_size * i) - len(compressed_dim) - 1), 'constant')
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
            for i in range(len(self.t_list)):
                t = self.t_list[i]
                while index0 < len(t_list0) - 1 and t_list0[index0 + 1] - t < -1e-4:
                    index0 = index0 + 1
                while index1 < len(t_list1) and t_list1[index1] - t < -1e-4:
                    index1 = index1 + 1
                    
                if index0 == len(t_list0) - 1:
                    temp_point = decompress_traj0[index0]
                else:
                    temp_point = decompress_traj0[index0] + (decompress_traj0[index0 + 1] - decompress_traj0[index0]) * (t - t_list0[index0]) / (t_list0[index0 + 1] - t_list0[index0])
                
                if index1 != len(t_list1) and abs(t_list1[index1] - t) < 1e-4:
                    self.decompress_traj.append(decompress_traj1[index1])
                else:
                    self.decompress_traj.append(temp_point)
        else:
            self.decompress_traj = decompress_traj1
        self.decompress_traj = np.array(self.decompress_traj)

class CISEDSCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', time_unit=0.1, drop=0):
        super().__init__('cised_s', path, data_source, time_unit, drop)
        pass
    
    @profile
    def compress(self, save_path=None, save=True, **kwargs):
        if save_path is None:
            save_path = self.save_path
            
        max_error = kwargs['max_error'] if 'max_error' in kwargs else 0.3
        polygon_num = kwargs['polygon_num'] if 'polygon_num' in kwargs else 16
        
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
                # write_bits = bitarray()
                
                # for i in range(0, len(index_list)):
                #     if i == 0 or \
                #         self.traj[index_list[i]][0] - self.traj[index_list[i - 1]][0] > 1e5 or \
                #         self.traj[index_list[i]][1] - self.traj[index_list[i - 1]][1] > 1e5 or \
                #         self.t_list[index_list[i]] - self.t_list[index_list[i - 1]] > 1e5:
                #         write_bits.extend(bitarray('0'))
                #         write_bits.extend(double2bitarray(self.t_list[index_list[i]]))
                #         write_bits.extend(double2bitarray(self.traj[index_list[i]][0]))
                #         write_bits.extend(double2bitarray(self.traj[index_list[i]][1]))
                #     else:
                #         write_bits.extend(bitarray('1'))
                #         write_bits.extend(float2bitarray(self.t_list[index_list[i]] - self.t_list[index_list[i - 1]]))
                #         write_bits.extend(float2bitarray(self.traj[index_list[i]][0] - self.traj[index_list[i - 1]][0]))
                #         write_bits.extend(float2bitarray(self.traj[index_list[i]][1] - self.traj[index_list[i - 1]][1]))
                # self.compress_bits = len(write_bits)
                # write_bits.tofile(f)
                
                
                write_bits = bitarray()
                
                for i in range(0, len(index_list)):
                    if i == 0 or \
                        self.traj[index_list[i]][0] - self.traj[index_list[last_index]][0] > 1e5 or \
                        self.traj[index_list[i]][1] - self.traj[index_list[last_index]][1] > 1e5 or \
                        self.t_list[index_list[i]] - self.t_list[index_list[last_index]] > 1e5:
                        write_bits.extend(bitarray('0'))
                        write_bits.extend(double2bitarray(self.t_list[index_list[i]]))
                        write_bits.extend(double2bitarray(self.traj[index_list[i]][0]))
                        write_bits.extend(double2bitarray(self.traj[index_list[i]][1]))
                        
                        last_index = i
                    else:
                        write_bits.extend(bitarray('1'))
                        write_bits.extend(float2bitarray(self.t_list[index_list[i]] - self.t_list[index_list[last_index]]))
                        write_bits.extend(float2bitarray(self.traj[index_list[i]][0] - self.traj[index_list[last_index]][0]))
                        write_bits.extend(float2bitarray(self.traj[index_list[i]][1] - self.traj[index_list[last_index]][1]))
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            read_index = 0
            
            t_list = []
            point_list = []
            while read_index < len(read_bits) - 8:
                type = bitarray2signint(read_bits[read_index:read_index+1], False)
                read_index += 1
                if type == 0:
                    t = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    x = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    y = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    t_list.append(t)
                    point_list.append([x, y])
                    last_t = t
                    last_x = x
                    last_y = y
                else:
                    t = bitarray2float(read_bits[read_index:read_index+32])
                    read_index += 32
                    x = bitarray2float(read_bits[read_index:read_index+32])
                    read_index += 32
                    y = bitarray2float(read_bits[read_index:read_index+32])
                    read_index += 32
                    # t_list.append(t + t_list[-1])
                    # point_list.append([x + point_list[-1][0], y + point_list[-1][1]])
                    t_list.append(t + last_t)
                    point_list.append([x + last_x, y + last_y])
        
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
        
class CISEDWCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', time_unit=0.1, drop=0):
        super().__init__('cised_w', path, data_source, time_unit, drop)
        pass
    
    @profile
    def compress(self, save_path=None, save=True, **kwargs):
        if save_path is None:
            save_path = self.save_path
            
        max_error = kwargs['max_error'] if 'max_error' in kwargs else 0.3
        polygon_num = kwargs['polygon_num'] if 'polygon_num' in kwargs else 16
        
        from shapely.geometry import Polygon, Point
        
        point_list = [self.traj[0]]
        id_list = [0]
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
            
            temp_polygon = get_polygon(point_list[-1], self.traj[i], id_list[-1], i, max_error)
            if polygon.is_empty:
                polygon = temp_polygon
            else:
                center_point = np.array([polygon.centroid.x, polygon.centroid.y])
                new_polygon = polygon.intersection(temp_polygon)
                if new_polygon.is_empty:
                    c = (self.t_list[id_list[-1] + 1] - self.t_list[id_list[-1]]) / (self.t_list[i - 1] - self.t_list[id_list[-1]])
                    add_point = self.traj[i - 1]
                    temp_point = point_list[-1] + (add_point - point_list[-1]) * c
                    if not polygon.contains(Point(temp_point)):
                        add_point = point_list[-1] + (center_point - point_list[-1]) / c
                    
                    point_list.append(add_point)
                    id_list.append(i - 1)
                    polygon = get_polygon(add_point, self.traj[i], id_list[-1], i, max_error)
                else:
                    polygon = new_polygon
        
        add_point = self.traj[-1]
        if not polygon.is_empty:
            center_point = np.array([polygon.centroid.x, polygon.centroid.y])
            c = (self.t_list[id_list[-1] + 1] - self.t_list[id_list[-1]]) / (self.t_list[len(self.traj) - 1] - self.t_list[id_list[-1]])
            temp_point = point_list[-1] + (add_point - point_list[-1]) * c
            if not polygon.contains(Point(temp_point)):
                add_point = point_list[-1] + (center_point - point_list[-1]) / c
        
        point_list.append(add_point)
        id_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                # write_bits = bitarray()
                # for i in range(0, len(id_list)):
                #     if i == 0 or \
                #         point_list[i][0] - point_list[i - 1][0] > 1e5 or \
                #         point_list[i][1] - point_list[i - 1][1] > 1e5 or \
                #         self.t_list[id_list[i]] - self.t_list[id_list[i - 1]] > 1e5:
                #         write_bits.extend(bitarray('0'))
                #         write_bits.extend(double2bitarray(self.t_list[id_list[i]]))
                #         write_bits.extend(double2bitarray(point_list[i][0]))
                #         write_bits.extend(double2bitarray(point_list[i][1]))
                #     else:
                #         write_bits.extend(bitarray('1'))
                #         write_bits.extend(float2bitarray(self.t_list[id_list[i]] - self.t_list[id_list[i - 1]]))
                #         write_bits.extend(float2bitarray(point_list[i][0] - point_list[i - 1][0]))
                #         write_bits.extend(float2bitarray(point_list[i][1] - point_list[i - 1][1]))
                # self.compress_bits = len(write_bits)
                # write_bits.tofile(f)
                
                
                write_bits = bitarray()
                for i in range(0, len(id_list)):
                    if i == 0 or \
                        point_list[i][0] - point_list[last_index][0] > 1e5 or \
                        point_list[i][1] - point_list[last_index][1] > 1e5 or \
                        self.t_list[id_list[i]] - self.t_list[id_list[last_index]] > 1e5:
                        write_bits.extend(bitarray('0'))
                        write_bits.extend(double2bitarray(self.t_list[id_list[i]]))
                        write_bits.extend(double2bitarray(point_list[i][0]))
                        write_bits.extend(double2bitarray(point_list[i][1]))
                        
                        last_index = i
                    else:
                        write_bits.extend(bitarray('1'))
                        write_bits.extend(float2bitarray(self.t_list[id_list[i]] - self.t_list[id_list[last_index]]))
                        write_bits.extend(float2bitarray(point_list[i][0] - point_list[last_index][0]))
                        write_bits.extend(float2bitarray(point_list[i][1] - point_list[last_index][1]))
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            read_index = 0
            
            t_list = []
            point_list = []
            while read_index < len(read_bits) - 8:
                type = bitarray2signint(read_bits[read_index:read_index+1], False)
                read_index += 1
                if type == 0:
                    t = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    x = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    y = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    t_list.append(t)
                    point_list.append([x, y])
                    last_t = t
                    last_x = x
                    last_y = y
                else:
                    t = bitarray2float(read_bits[read_index:read_index+32])
                    read_index += 32
                    x = bitarray2float(read_bits[read_index:read_index+32])
                    read_index += 32
                    y = bitarray2float(read_bits[read_index:read_index+32])
                    read_index += 32
                    # t_list.append(t + t_list[-1])
                    # point_list.append([x + point_list[-1][0], y + point_list[-1][1]])
                    t_list.append(t + last_t)
                    point_list.append([x + last_x, y + last_y])
        
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
        
class SQUISHCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', time_unit=0.1, drop=0):
        super().__init__('squish', path, data_source, time_unit, drop)
        pass
    
    @profile
    def compress(self, save_path=None, save=True, **kwargs):
        if save_path is None:
            save_path = self.save_path
            
        max_error = kwargs['max_error'] if 'max_error' in kwargs else 0.3
        compression_ratio = kwargs['compression_ratio'] if 'compression_ratio' in kwargs else 1
        
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
                traj_dim = self.traj.shape[1]
                for i in range(0, len(index_list)):
                    write_bits.extend(double2bitarray(self.t_list[index_list[i]]))
                    for j in range(traj_dim):
                        write_bits.extend(double2bitarray(self.traj[index_list[i]][j]))
                        
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            read_index = 0
            
            traj_dim = self.traj.shape[1]
            t_list = []
            point_list = []
            while read_index < len(read_bits) - 8:
                t = bitarray2double(read_bits[read_index:read_index+64])
                read_index += 64
                point = []
                for j in range(traj_dim):
                    x = bitarray2double(read_bits[read_index:read_index+64])
                    read_index += 64
                    point.append(x)
                t_list.append(t)
                point_list.append(point)
        
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
                    

import multiprocessing

def process_file(file_path,
                 data_source,
                 time_unit,
                 drop,
                 dct_no_map_block_size,
                 dct_no_map_compression_ratio,
                 dct_no_map_scale,
                 dct_no_map_scale_length,
                 dct_dynamic_ratio_flag,
                 cised_max_error):
    logs = f'{file_path}\n'
    
    result = []
    
    # squish_compressor = SQUISHCompressor(file_path, data_source, time_unit, drop)
    # squish_compressor.compress(max_error=cised_max_error)
    # squish_compressor.decompress()
    
    # squish_result = squish_compressor.evaluation_metrics(cised_max_error)
    # # squish_compressor.plot_traj()
    # logs += f"squish performance: {squish_result}\n"
    # result.append([squish_compressor.name, squish_result])
    
    # dct_no_map_compressor = DCTNoMapCompressor(file_path, data_source, time_unit, drop)
    # # dct_no_map_compressor.compress(block_size=dct_no_map_block_size, compression_ratio=dct_no_map_compression_ratio, scale=dct_no_map_scale, dynamic_ratio_flag=dct_dynamic_ratio_flag, scale_length=dct_no_map_scale_length, max_error=cised_max_error)
    # dct_no_map_compressor.compress(block_size=dct_no_map_block_size, compression_ratio=dct_no_map_compression_ratio, scale=dct_no_map_scale, dynamic_ratio_flag=dct_dynamic_ratio_flag, scale_length=dct_no_map_scale_length,)
    # dct_no_map_compressor.decompress()
    
    # dct_no_map_result = dct_no_map_compressor.evaluation_metrics(cised_max_error)
    # # dct_no_map_compressor.plot_traj()
    # logs += f"dct no map performance: {dct_no_map_result}\n"
    # result.append([dct_no_map_compressor.name, dct_no_map_result])
    
    ciseds_compressor = CISEDSCompressor(file_path, data_source, time_unit, drop)
    ciseds_compressor.compress(max_error=cised_max_error)
    ciseds_compressor.decompress()
    
    ciseds_result = ciseds_compressor.evaluation_metrics(cised_max_error)
    # ciseds_compressor.plot_traj(dct_no_map_compressor.argmax_error)
    logs += f"cised s performance: {ciseds_result}\n"
    result.append([ciseds_compressor.name, ciseds_result])
    
    cisedw_compressor = CISEDWCompressor(file_path, data_source, time_unit, drop)
    cisedw_compressor.compress(max_error=cised_max_error)
    cisedw_compressor.decompress()
    
    cisedw_result = cisedw_compressor.evaluation_metrics(cised_max_error)
    # cisedw_compressor.plot_traj(dct_no_map_compressor.argmax_error)
    logs += f"cised w performance: {cisedw_result}\n"
    result.append([cisedw_compressor.name, cisedw_result])
    
    # print(logs, flush=True)
    logs += '\n'
    return result, logs

def main():
    data_source = 'mopsi'
    
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
    drop = 0
    
    with open(f'config/{data_source}.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    time_unit = config['time_unit']
    dct_no_map_block_size = config[f'drop_{drop}']['dct_no_map']['block_size']
    dct_no_map_compression_ratio = config[f'drop_{drop}']['dct_no_map']['compression_ratio']
    dct_no_map_scale = config[f'drop_{drop}']['dct_no_map']['scale']
    dct_no_map_scale_length = config[f'drop_{drop}']['dct_no_map']['scale_length']
    dct_dynamic_ratio_flag = config[f'drop_{drop}']['dct_no_map']['dynamic_ratio_flag']
    cised_max_error = config[f'drop_{drop}']['cised']['max_error']
    
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
    
    global_counter = Global_Counter()
    results = []
    logs = ''
    
    if multiprocessing_flag:
        with multiprocessing.Pool(processes=16) as pool:
            pool_results = []
            for ff in tot_path:
                result = pool.apply_async(process_file, (ff,
                                data_source,
                                time_unit,
                                drop,
                                dct_no_map_block_size,
                                dct_no_map_compression_ratio,
                                dct_no_map_scale,
                                dct_no_map_scale_length,
                                dct_dynamic_ratio_flag,
                                cised_max_error))
                pool_results.append(result)
            
            for result in pool_results:
                res, log = result.get()
                results.extend(res)
                logs += log
    else:
        for ff in tot_path:
            if ff not in ['/home/wkf/data/PRESS/datasets/geolife/traj_065.txt',
                          ]:
                continue
            
            result, log = process_file(ff,
                                data_source,
                                time_unit,
                                drop,
                                dct_no_map_block_size,
                                dct_no_map_compression_ratio,
                                dct_no_map_scale,
                                dct_dynamic_ratio_flag,
                                cised_max_error)
            results.extend(result)
            logs += log
        
    for r in results:
        global_counter.count(r[0], r[1])
    
    print(logs)
    global_counter.get_result()

if __name__ == '__main__':
    main()