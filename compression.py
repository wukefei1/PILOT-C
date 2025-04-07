import numpy as np
import math
import os
import yaml
from scipy.fft import *
import matplotlib.pyplot as plt
from nuplan_map import *
from utils import *
import struct
import pickle
from bitarray import bitarray

if 'LINE_PROFILE_IT' not in os.environ:
    def profile(x):
        return x

def lossless_compress(compressed_result, bits = None):
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
    
    dp = np.full((len(compressed_result) + 1, max_bits), np.inf)
    lossless_compressed_result = np.zeros((len(compressed_result) + 1, max_bits))
    
    for j in range(max_bits):
        dp[0][j] = 5 + 8 + j
    
    for i in range(len(compressed_result)):
        argmin_dp_i = np.argmin(dp[i])
        for j in range(need_bits[i], max_bits):
            if dp[i][argmin_dp_i] + 5 + 8 + j > dp[i][j] + j:
                dp[i + 1][j] = dp[i][j] + j
                lossless_compressed_result[i + 1][j] = j
            else:
                dp[i + 1][j] = dp[i][argmin_dp_i] + 5 + 8 + j
                lossless_compressed_result[i + 1][j] = argmin_dp_i
    
    if bits is None:
        end_index = len(compressed_result)
    else:
        end_index = 0
        for i in range(len(compressed_result), 0, -1):
            if np.min(dp[i]) <= bits:
                end_index = i
                break
    
    flag = 0
    argmin_dp = int(np.argmin(dp[end_index]))
    temp_result = []
    for i in range(end_index, 0, -1):
        if argmin_dp != 0:
            flag = 1
        else:
            end_index = end_index - 1
        if flag == 1:
            temp_result.append(argmin_dp)
        argmin_dp = int(lossless_compressed_result[i][argmin_dp])
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

class Compressor:
    '''
    base class
    '''
    _cache_version = 'v20250113'
    _cache_attributes = {
        'map',
        'traj',
        'lc_list',
        'length_list',
        'dist_list',
    }
    
    def __init__(self, name, path, maps=None, data_source='nuplan', dist_delta=1):
        self.dist_delta = dist_delta
        if data_source == 'nuplan':
            assert maps is not None, "maps should not be None when data_source is nuplan"
            path_name = path.split("/")[-1]
            self.name = name
            self.path = path
            self.cache_path = f'cache/nuplan/{path_name}/'
            self.result_path = f'result/nuplan/{path_name}/'
            self.save_path = self.result_path + f'compressed_{name}'
            self.load_path = self.result_path + f'compressed_{name}'
            self.origin_path = self.result_path + f'origin'
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
            
            self.load_nuplan_traj(maps)
        if data_source == 'geolife':
            path_name = path.split("/")[-1].split(".")[0]
            self.name = name
            self.path = path
            self.cache_path = f'cache/geolife/{path_name}/'
            self.result_path = f'result/geolife/{path_name}/'
            self.save_path = self.result_path + f'compressed_{name}'
            self.load_path = self.result_path + f'compressed_{name}'
            self.origin_path = self.result_path + f'origin'
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            if not os.path.exists(self.result_path):
                os.makedirs(self.result_path)
            
            self.load_geolife_traj()

    def compress(self, save_path=None, **kwargs):
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
    
    def load_geolife_traj(self):
        if not self._load_cache_file(os.path.join(self.cache_path, 'cache.pkl')):
            data = [x.strip().split(",") for x in open(self.path).readlines()]
            data = data[6:]
            traj = [[float(x[0]), float(x[1])] for x in data]
            traj = np.array(traj)
            self.traj = traj
            
            with open(self.origin_path, 'wb') as f:
                for x, y in self.traj:
                    f.write(struct.pack('dd', x, y))
            self.map = None
            self.lc_list = None
            self.length_list = None
            self.dist_list = None
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
    
    def load_nuplan_traj(self, maps):
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
            for line in data:
                x = get_data(line, title, "X_utm(m)")
                y = get_data(line, title, "Y_utm(m)")
                traj.append([float(x), float(y)])
            
            traj = np.array(traj)
            self.traj = traj
            
            with open(self.origin_path, 'wb') as f:
                for x, y in self.traj:
                    f.write(struct.pack('dd', x, y))
            
            if os.path.exists(os.path.join(self.path, 'meta.yml')):
                with open(os.path.join(self.path, 'meta.yml'), 'r') as file:
                    config = yaml.safe_load(file)
                    self.map = maps[config['map']]
            else:
                raise FileNotFoundError('meta.yml not found')
            
            result = self.map.get_data_from_trajectory(self.traj)
            
            # get lc list
            # get length list
            
            lc_list = []
            length_list_raw = []
            dist_list_raw = []
            
            last_lc_id = -1
            last_is_virtual = False
            tot_length = 0
            tot_dist = 0
            for i in range(len(result)):
                is_virtual = result[i]['is_virtual']
                dist = result[i]['dist']
                length = result[i]['len']
                if is_virtual:
                    if last_lc_id != result[i]["LCC_id"]:
                        lc_list.append([result[i]["LCC_id"], 1, i])
                        if last_lc_id != -1:
                            geo_point = self.map.lanecenterlineconnectors[result[i]["LCC_id"]]["geometry"][0]
                            temp_result = self.map._get_data_from_location_and_LCC_id(geo_point[0], geo_point[1], last_lc_id) if last_is_virtual else self.map._get_data_from_location_and_LC_id(geo_point[0], geo_point[1], last_lc_id)
                            tot_length = tot_length + temp_result['len']
                            tot_dist = tot_dist + temp_result['dist']
                        last_lc_id = result[i]["LCC_id"]
                        last_is_virtual = is_virtual
                else:
                    if last_lc_id != result[i]["LC_id"]:
                        lc_list.append([result[i]["LC_id"], 0, i])
                        if last_lc_id != -1:
                            geo_point = self.map.lanecenterlines[result[i]["LC_id"]]["geometry"][0]
                            temp_result = self.map._get_data_from_location_and_LCC_id(geo_point[0], geo_point[1], last_lc_id) if last_is_virtual else self.map._get_data_from_location_and_LC_id(geo_point[0], geo_point[1], last_lc_id)
                            tot_length = tot_length + temp_result['len']
                            tot_dist = tot_dist + temp_result['dist']
                        last_lc_id = result[i]["LC_id"]
                        last_is_virtual = is_virtual
                length_list_raw.append(length + tot_length)
                dist_list_raw.append(dist + tot_dist)
                
            self.lc_list = lc_list
            
            length_list = np.array(length_list_raw)
            self.length_list = length_list
            
            # get dist list
            dist_list = []
            temp_index = 1
            for i in range(int(min(length_list) / self.dist_delta), int(max(length_list) / self.dist_delta) + 2):
                while length_list[temp_index] <= i * self.dist_delta and temp_index < len(length_list) - 1:
                    temp_index += 1
                temp_dist = dist_list_raw[temp_index - 1] + (i * self.dist_delta - length_list[temp_index - 1]) * (dist_list_raw[temp_index] - dist_list_raw[temp_index - 1]) / (length_list[temp_index] - length_list[temp_index - 1])
                dist_list.append(temp_dist)
            
            dist_list = np.array(dist_list)
            self.dist_list = dist_list
            
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
    
    def get_decode_traj(self, decompressed_length, decompressed_dist, lc_list):
        '''
        get decode traj from decompressed_length, decompressed_dist, lc_list
        '''
        decode_list = []
        tot_length = 0
        tot_dist = 0
        for i in range(len(lc_list)):
            index0 = lc_list[i][2]
            index1 = lc_list[i + 1][2] if i < len(lc_list) - 1 else len(decompressed_length)
            
            if i != 0:
                geo_point = self.map.lanecenterlineconnectors[lc_list[i][0]]["geometry"][0] if lc_list[i][1] == 1 else self.map.lanecenterlines[lc_list[i][0]]["geometry"][0]
                temp_result = self.map._get_data_from_location_and_LCC_id(geo_point[0], geo_point[1], lc_list[i - 1][0]) if lc_list[i - 1][1] == 1 else self.map._get_data_from_location_and_LC_id(geo_point[0], geo_point[1], self.lc_list[i - 1][0])
                tot_length = tot_length + temp_result['len']
                tot_dist = tot_dist + temp_result['dist']
            for j in range(index0, index1):
                dist_index = min(len(decompressed_dist) - 2, int(decompressed_length[j] / self.dist_delta) - int(min(decompressed_length) / self.dist_delta))
                temp_dist = decompressed_dist[dist_index] + (decompressed_length[j] - dist_index * self.dist_delta - int(min(decompressed_length))) * (decompressed_dist[dist_index + 1] - decompressed_dist[dist_index]) / self.dist_delta - tot_dist
                # temp_dist = 0
                temp_length = decompressed_length[j] - tot_length
                if lc_list[i][1] == 1:
                    point = self.map.get_point_from_data_lcc(lc_list[i][0], temp_dist, temp_length)
                if lc_list[i][1] == 0:
                    point = self.map.get_point_from_data_lc(lc_list[i][0], temp_dist, temp_length)
                decode_list.append(point)
                
        self.decompress_traj = np.array(decode_list)

    def evaluation_metrics(self):
        '''
        calculate max error, mean error, compress ratio
        '''
        max_error = 0
        mean_error = 0
        for i in range(len(self.traj)):
            max_error = max(max_error, np.sqrt(np.sum((self.traj[i] - self.decompress_traj[i]) ** 2)))
            mean_error += np.sqrt(np.sum((self.traj[i] - self.decompress_traj[i]) ** 2))
        mean_error = mean_error / len(self.traj)
        return {'max_error': float(max_error),
                'mean_error': float(mean_error),
                'compress_ratio': float(len(self.traj) * 16 / self.compress_bits * 8)}
    
    def get_origin_metrics(self):
        self.get_decode_traj(self.length_list, self.dist_list, self.lc_list)
        self.compress_bits = len(self.traj) * 128
        return self.evaluation_metrics()

class SimpleCompressor(Compressor):
    def __init__(self, path, maps):
        super().__init__('simple', path, maps)
        pass

    def compress(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = self.save_path
        
        bits = kwargs['bits']
        
        divide = 2 ** (bits - 1) - 1
        start_length = self.length_list[0]
        delta_length = np.abs(np.array(self.length_list[1:] - self.length_list[:-1])).max() / divide
        simple_length = []
        tot_compress_length = start_length
        for i in range(1, len(self.length_list)):
            temp = np.round((self.length_list[i] - tot_compress_length) / delta_length)
            simple_length.append(int(temp))
            tot_compress_length += temp * delta_length
        
        start_dist = self.dist_list[0]
        delta_dist = np.abs(np.array(self.dist_list[1:] - self.dist_list[:-1])).max() / divide
        simple_dist = []
        tot_compress_dist = start_dist
        for i in range(1, len(self.dist_list)):
            temp = np.round((self.dist_list[i] - tot_compress_dist) / delta_dist)
            simple_dist.append(int(temp))
            tot_compress_dist += temp * delta_dist
        
        with open(save_path, 'wb') as f:
            write_bits = bitarray()
            write_bits.extend(signint2bitarray(bits, 4))
            write_bits.extend(double2bitarray(start_length))
            write_bits.extend(double2bitarray(delta_length))
            write_bits.extend(signint2bitarray(len(simple_length), 32))
            for i in range(len(simple_length)):
                temp = simple_length[i]
                write_bits.extend(signint2bitarray(temp, bits))
            write_bits.extend(double2bitarray(start_dist))
            write_bits.extend(double2bitarray(delta_dist))
            write_bits.extend(signint2bitarray(len(simple_dist), 32))
            for i in range(len(simple_dist)):
                temp = simple_dist[i]
                if temp < 0:
                    temp = temp + 2 ** bits
                write_bits.extend(signint2bitarray(temp, bits))
            
            write_bits.extend(signint2bitarray(len(self.lc_list), 32))
            for i in range(len(self.lc_list)):
                write_bits.extend(signint2bitarray(self.lc_list[i][0], 32))
                write_bits.extend(signint2bitarray(self.lc_list[i][1], 1))
                write_bits.extend(signint2bitarray(self.lc_list[i][2], 32))
            
            self.compress_bits = len(write_bits)
            write_bits.tofile(f)
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            bits = bitarray2signint(read_bits[:4], False)
            read_bits = read_bits[4:]
            start_length = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            delta_length = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            simple_length = []
            for i in range(length_num):
                temp = bitarray2signint(read_bits[:bits])
                if temp >= 2 ** (bits - 1):
                    temp = temp - 2 ** bits
                simple_length.append(temp)
                read_bits = read_bits[bits:]
            start_dist = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            delta_dist = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            dist_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            simple_dist = []
            for i in range(dist_num):
                temp = bitarray2signint(read_bits[:bits])
                if temp >= 2 ** (bits - 1):
                    temp = temp - 2 ** bits
                simple_dist.append(temp)
                read_bits = read_bits[bits:]
            
            lc_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            lc_list = []
            for i in range(lc_num):
                lc_id = bitarray2signint(read_bits[:32], False)
                is_virtual = bitarray2signint(read_bits[32:33], False)
                index = bitarray2signint(read_bits[33:65], False)
                lc_list.append([lc_id, is_virtual, index])
                read_bits = read_bits[65:]
        
        decompress_length = [start_length]
        for i in range(len(simple_length)):
            decompress_length.append(decompress_length[-1] + simple_length[i] * delta_length)
        
        decompress_dist = [start_dist]
        for i in range(len(simple_dist)):
            decompress_dist.append(decompress_dist[-1] + simple_dist[i] * delta_dist)
        
        self.get_decode_traj(decompress_length, decompress_dist, lc_list)

class PressCompressor(Compressor):
    def __init__(self, path, maps):
        super().__init__('press', path, maps)
        pass
    
    def compress(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = self.save_path
        
        max_error_length = kwargs['max_error_length'] if 'max_error_length' in kwargs else 0.2
        max_error_dist = kwargs['max_error_dist'] if 'max_error_dist' in kwargs else 0.1
        
        index = 0
        press_length = [[float(self.length_list[index]), index]]
        r = [-np.pi / 2, np.pi / 2]
        for i in range(1, len(self.length_list)):
            r_i = math.atan2(self.length_list[i] - self.length_list[index], i - index)
            if r[0] <= r_i and r_i <= r[1]:
                temp_r = [math.atan2(self.length_list[i] - self.length_list[index] - max_error_length, i - index), math.atan2(self.length_list[i] - self.length_list[index] + max_error_length, i - index)]
                r = [max(r[0], temp_r[0]), min(r[1], temp_r[1])]
            else:
                index = i - 1
                press_length.append([float(self.length_list[index]), index])
                r = [-np.pi / 2, np.pi / 2]
        press_length.append([float(self.length_list[-1]), len(self.length_list) - 1])
        
        index = 0
        press_dist = [[float(self.dist_list[index]), index]]
        r = [-np.pi / 2, np.pi / 2]
        for i in range(1, len(self.dist_list)):
            r_i = math.atan2(self.dist_list[i] - self.dist_list[index], i - index)
            if r[0] <= r_i and r_i <= r[1]:
                temp_r = [math.atan2(self.dist_list[i] - self.dist_list[index] - max_error_dist, i - index), math.atan2(self.dist_list[i] - self.dist_list[index] + max_error_dist, i - index)]
                r = [max(r[0], temp_r[0]), min(r[1], temp_r[1])]
            else:
                index = i - 1
                press_dist.append([float(self.dist_list[index]), index])
                r = [-np.pi / 2, np.pi / 2]
        press_dist.append([float(self.dist_list[-1]), len(self.dist_list) - 1])
        
        with open(save_path, 'wb') as f:
            write_bits = bitarray()
            write_bits.extend(signint2bitarray(len(press_length), 32))
            for i in range(len(press_length)):
                write_bits.extend(float2bitarray(press_length[i][0]))
                write_bits.extend(signint2bitarray(press_length[i][1], 32))
            write_bits.extend(signint2bitarray(len(press_dist), 32))
            for i in range(len(press_dist)):
                write_bits.extend(float2bitarray(press_dist[i][0]))
                write_bits.extend(signint2bitarray(press_dist[i][1], 32))
            
            write_bits.extend(signint2bitarray(len(self.lc_list), 32))
            for i in range(len(self.lc_list)):
                write_bits.extend(signint2bitarray(self.lc_list[i][0], 32))
                write_bits.extend(signint2bitarray(self.lc_list[i][1], 1))
                write_bits.extend(signint2bitarray(self.lc_list[i][2], 32))
            
            self.compress_bits = len(write_bits)
            write_bits.tofile(f)
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
            
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            press_length = []
            for i in range(length_num):
                press_length.append([bitarray2float(read_bits[:32]), bitarray2signint(read_bits[32:64])])
                read_bits = read_bits[64:]
            dist_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            press_dist = []
            for i in range(dist_num):
                press_dist.append([bitarray2float(read_bits[:32]), bitarray2signint(read_bits[32:64])])
                read_bits = read_bits[64:]
            
            lc_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            lc_list = []
            for i in range(lc_num):
                lc_id = bitarray2signint(read_bits[:32], False)
                is_virtual = bitarray2signint(read_bits[32:33], False)
                index = bitarray2signint(read_bits[33:65], False)
                lc_list.append([lc_id, is_virtual, index])
                read_bits = read_bits[65:]
        
        decompress_length = []
        decompress_dist = []
        for i in range(len(press_length)):
            if i == len(press_length) - 1:
                decompress_length.append(press_length[i][0])
            else:
                for j in range(press_length[i][1], press_length[i + 1][1]):
                    temp = (j - press_length[i][1]) / (press_length[i + 1][1] - press_length[i][1]) * (press_length[i + 1][0] - press_length[i][0]) + press_length[i][0]
                    decompress_length.append(temp)
        for i in range(len(press_dist)):
            if i == len(press_dist) - 1:
                decompress_dist.append(press_dist[i][0])
            else:
                for j in range(press_dist[i][1], press_dist[i + 1][1]):
                    temp = (j - press_dist[i][1]) / (press_dist[i + 1][1] - press_dist[i][1]) * (press_dist[i + 1][0] - press_dist[i][0]) + press_dist[i][0]
                    decompress_dist.append(temp)
        
        self.get_decode_traj(decompress_length, decompress_dist, lc_list)

class DCTCompressor(Compressor):
    def __init__(self, path, maps):
        super().__init__('dct', path, maps)
        
        length_list = self.length_list[::10]
        for i in range(1, len(length_list)):
            if length_list[i] < length_list[i - 1]:
                length_list[i] = length_list[i - 1]
        
        print(len(length_list))
        for i in range(len(length_list)):
            print(i, length_list[i])
            
        # # for i in range(len(self.length_list) - 1, -1, -1):
        # #     print(i * 0.1, self.length_list[i] + 0.1)
        pass
    
    def compress(self, save_path=None, **kwargs):
        if save_path is None:
            save_path = self.save_path
        
        percent_length = kwargs['percent_length'] if 'percent_length' in kwargs else 0.3
        percent_dist = kwargs['percent_dist'] if 'percent_dist' in kwargs else 0.3
        block_size = kwargs['block_size'] if 'block_size' in kwargs else 640
        max_compressed_block_size = kwargs['max_compressed_block_size'] if'max_compressed_block_size' in kwargs else 255
        scale_length = kwargs['scale_length'] if 'scale_length' in kwargs else 2
        scale_dist = kwargs['scale_dist'] if 'scale_dist' in kwargs else 2
        
        length_list = np.array(self.length_list)
        delta_length = length_list[1:] - length_list[:-1]
        
        result_length = []
        for block_id in range(0, (len(delta_length) - 1) // block_size + 1):
            block_length = delta_length[block_id * block_size: min((block_id + 1) * block_size, len(delta_length))]
            mean = np.mean(block_length)
            block_length = block_length - mean
            
            dctn_length = dctn(block_length)
            compressed_length = []
            index = min(max_compressed_block_size, len(dctn_length) - 1)
            while int(np.round(dctn_length[index] * scale_length)) == 0 and index > 0:
                index = index - 1
            
            for i in range(1, index + 1):
                compressed_length.append(int(np.round(dctn_length[i] * scale_length)))
            
            lossless_compressed_length = lossless_compress(compressed_length)
            result_length.append([lossless_compressed_length, mean])
        
        dist_list = np.array(self.dist_list)
        delta_dist = dist_list[1:] - dist_list[:-1]
        
        result_dist = []
        for block_id in range(0, (len(delta_dist) - 1) // block_size + 1):
            block_dist = delta_dist[block_id * block_size: min((block_id + 1) * block_size, len(delta_dist))]
            mean = np.mean(block_dist)
            block_dist = block_dist - mean
            
            dctn_dist = dctn(block_dist)
            compressed_dist = []
            index = min(max_compressed_block_size, len(dctn_dist) - 1)
            while int(np.round(dctn_dist[index] * scale_dist)) == 0 and index > 0:
                index = index - 1
            
            for i in range(1, index + 1):
                compressed_dist.append(int(np.round(dctn_dist[i] * scale_dist)))
            
            lossless_compressed_dist = lossless_compress(compressed_dist)
            result_dist.append([lossless_compressed_dist, mean])
        
        with open(save_path, 'wb') as f:
            write_bits = bitarray()
            write_bits.extend(signint2bitarray(block_size, 32))
            write_bits.extend(double2bitarray(scale_length))
            write_bits.extend(double2bitarray(scale_dist))
            
            write_bits.extend(double2bitarray(length_list[0]))
            write_bits.extend(signint2bitarray(len(delta_length), 32))
            write_bits.extend(signint2bitarray(len(result_length), 32))
            for i in range(len(result_length)):
                write_bits.extend(signint2bitarray(len(result_length[i][0]), 8))
                for j in range(len(result_length[i][0])):
                    write_bits.extend(signint2bitarray(result_length[i][0][j][0], 4))
                    write_bits.extend(signint2bitarray(len(result_length[i][0][j][1]), 8))
                    for k in range(len(result_length[i][0][j][1])):
                        write_bits.extend(signint2bitarray(result_length[i][0][j][1][k], result_length[i][0][j][0]))
                write_bits.extend(double2bitarray(result_length[i][1]))
            
            write_bits.extend(double2bitarray(dist_list[0]))
            write_bits.extend(signint2bitarray(len(delta_dist), 32))
            write_bits.extend(signint2bitarray(len(result_dist), 32))
            for i in range(len(result_dist)):
                write_bits.extend(signint2bitarray(len(result_dist[i][0]), 8))
                for j in range(len(result_dist[i][0])):
                    write_bits.extend(signint2bitarray(result_dist[i][0][j][0], 4))
                    write_bits.extend(signint2bitarray(len(result_dist[i][0][j][1]), 8))
                    for k in range(len(result_dist[i][0][j][1])):
                        write_bits.extend(signint2bitarray(result_dist[i][0][j][1][k], result_dist[i][0][j][0]))
                write_bits.extend(double2bitarray(result_dist[i][1]))
            
            write_bits.extend(signint2bitarray(len(self.lc_list), 32))
            for i in range(len(self.lc_list)):
                write_bits.extend(signint2bitarray(self.lc_list[i][0], 32))
                write_bits.extend(signint2bitarray(self.lc_list[i][1], 1))
                write_bits.extend(signint2bitarray(self.lc_list[i][2], 32))
            
            self.compress_bits = len(write_bits)
            write_bits.tofile(f)
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            block_size = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            scale_length = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            scale_dist = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            
            start_length = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            delta_length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            result_length = []
            for i in range(length_num):
                length_block_num = bitarray2signint(read_bits[:8], False)
                read_bits = read_bits[8:]
                compressed_length = []
                for j in range(length_block_num):
                    bits = bitarray2signint(read_bits[:4], False)
                    read_bits = read_bits[4:]
                    temp_length_num = bitarray2signint(read_bits[:8], False)
                    read_bits = read_bits[8:]
                    for k in range(temp_length_num):
                        compressed_length.append(bitarray2signint(read_bits[:bits]))
                        read_bits = read_bits[bits:]
                mean = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                result_length.append([compressed_length, mean])
            
            start_dist = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            delta_dist_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            dist_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            result_dist = []
            for i in range(dist_num):
                dist_block_num = bitarray2signint(read_bits[:8], False)
                read_bits = read_bits[8:]
                compressed_dist = []
                for j in range(dist_block_num):
                    bits = bitarray2signint(read_bits[:4], False)
                    read_bits = read_bits[4:]
                    temp_dist_num = bitarray2signint(read_bits[:8], False)
                    read_bits = read_bits[8:]
                    for k in range(temp_dist_num):
                        compressed_dist.append(bitarray2signint(read_bits[:bits]))
                        read_bits = read_bits[bits:]
                mean = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                result_dist.append([compressed_dist, mean])
            
            lc_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            lc_list = []
            for i in range(lc_num):
                lc_id = bitarray2signint(read_bits[:32], False)
                is_virtual = bitarray2signint(read_bits[32:33], False)
                index = bitarray2signint(read_bits[33:65], False)
                lc_list.append([lc_id, is_virtual, index])
                read_bits = read_bits[65:]
        
        tot_idct_length = np.array([])
        for i in range(len(result_length) - 1):
            compressed_length = result_length[i][0]
            mean = result_length[i][1]
            compressed_length = np.pad(compressed_length, (1, block_size - len(compressed_length) - 1), 'constant')
            idct_length = idctn(compressed_length / scale_length)
            idct_length = idct_length + mean
            tot_idct_length = np.concatenate((tot_idct_length, idct_length))
        compressed_length = result_length[-1][0]
        mean = result_length[-1][1]
        compressed_length = np.pad(compressed_length, (1, delta_length_num - block_size * (len(result_length) - 1) - len(compressed_length) - 1), 'constant')
        idct_length = idctn(compressed_length / scale_length)
        idct_length = idct_length + mean
        tot_idct_length = np.concatenate((tot_idct_length, idct_length))

        decompress_length = [start_length]
        for i in range(len(tot_idct_length)):
            decompress_length.append(decompress_length[i] + tot_idct_length[i])
        
        tot_idct_dist = np.array([])
        for i in range(len(result_dist) - 1):
            compressed_dist = result_dist[i][0]
            mean = result_dist[i][1]
            compressed_dist = np.pad(compressed_dist, (1, block_size - len(compressed_dist) - 1), 'constant')
            idct_dist = idctn(compressed_dist / scale_dist)
            idct_dist = idct_dist + mean
            tot_idct_dist = np.concatenate((tot_idct_dist, idct_dist))
        compressed_dist = result_dist[-1][0]
        mean = result_dist[-1][1]
        compressed_dist = np.pad(compressed_dist, (1, delta_dist_num - block_size * (len(result_dist) - 1) - len(compressed_dist) - 1), 'constant')
        idct_dist = idctn(compressed_dist / scale_dist)
        idct_dist = idct_dist + mean
        tot_idct_dist = np.concatenate((tot_idct_dist, idct_dist))

        decompress_dist = [start_dist]
        for i in range(len(tot_idct_dist)):
            decompress_dist.append(decompress_dist[i] + tot_idct_dist[i])
        
        self.get_decode_traj(decompress_length, decompress_dist, lc_list)
        
class NoMapCompressor():
    '''
    base class
    '''
    _cache_version = 'v20250205'
    _cache_attributes = {
        'traj',
    }
    
    def __init__(self, name, path, data_source='nuplan', delta_t: int=0.1):
        path_name = path.split("/")[-1]
        self.name = name
        self.path = path
        self.delta_t = delta_t
        self.cache_path = f'cache/{data_source}/delta_t_{delta_t}/{path_name}/'
        self.result_path = f'result/{data_source}/delta_t_{delta_t}/{path_name}/'
        self.save_path = self.result_path + f'compressed_{name}'
        self.load_path = self.result_path + f'compressed_{name}'
        self.origin_path = self.result_path + f'origin'
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.load_nuplan_traj()
        
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
    
    @profile
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
                traj.append([float(x), float(y)])
                # if len(t_list) > 0:
                #     assert abs(t - t_list[-1] - 0.1) < 0.2, "time not continuous"
                t_list.append(t)
            
            delta_t = (t_list[-1] - t_list[0]) / (len(t_list) - 1)
            frequency = int(np.round(self.delta_t / delta_t))
            
            traj = np.array(traj)
            self.traj = traj[::frequency]
            
            with open(self.origin_path, 'wb') as f:
                for x, y in self.traj:
                    f.write(struct.pack('dd', x, y))
            self._save_cache_file(os.path.join(self.cache_path, 'cache.pkl'))
            
    def evaluation_metrics(self):
        '''
        calculate max error, mean error, compress ratio
        '''
        error = np.sqrt(np.sum((self.traj - self.decompress_traj) ** 2, axis=1))
        max_error = np.max(error)
        mean_error = np.mean(error)
        self.argmax_error = int(np.argmax(error))
        return {'max_error': float(max_error),
                'mean_error': float(mean_error),
                'compress_ratio': float(len(self.traj) * 16 / self.compress_bits * 8),
                'length': len(self.traj)}
        
    def plot_traj(self, argmax_error=None):
        if self.decompress_traj is None:
            print("Please decompress first")
            return
        
        if argmax_error is None:
            plt.scatter(self.traj[:, 0], self.traj[:, 1], label='origin', s=0.01)
            plt.scatter(self.decompress_traj[:, 0], self.decompress_traj[:, 1], label='decompress', s=0.01)
            
        else:
            start_index = argmax_error - 10
            end_index = argmax_error + 10
            
            plt.scatter(self.traj[start_index: end_index, 0], self.traj[start_index: end_index, 1], label='origin', s=1)
            plt.scatter(self.decompress_traj[start_index: end_index, 0], self.decompress_traj[start_index: end_index, 1], label='decompress', s=1)
        plt.axis('equal')
        
        plt.legend()
        plt.savefig(f'images/test_{self.name}.png', dpi=500)
        plt.gcf().clear()

class DCTNoMapCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', delta_t=0.1):
        super().__init__('dct_no_map', path, data_source, delta_t)
        pass
    
    @profile
    def compress(self, save_path=None, save=True, **kwargs):
        if save_path is None:
            save_path = self.save_path
        
        block_size = kwargs['block_size'] if 'block_size' in kwargs else 300
        max_compressed_block_size = kwargs['max_compressed_block_size'] if 'max_compressed_block_size' in kwargs else int(min(60, 0.2 * len(self.traj)))
        compression_ratio = kwargs['compression_ratio'] if 'compression_ratio' in kwargs else None
        scale = kwargs['scale'] if 'scale' in kwargs else 1
        
        x_list = self.traj[:, 0]
        delta_x = x_list[1:] - x_list[:-1]
        y_list = self.traj[:, 1]
        delta_y = y_list[1:] - y_list[:-1]
        result_x = []
        result_y = []
        
        block_num = (len(delta_x) - 1) // block_size + 1
        block_size = (len(delta_x) - 1) // block_num + 1
        
        if compression_ratio is None:
            for block_id in range(0, (len(delta_x) - 1) // block_size + 1):
                block_x = delta_x[block_id * block_size: min((block_id + 1) * block_size, len(delta_x))]
                mean = np.mean(block_x)
                block_x = block_x - mean
                
                dctn_x = dctn(block_x)
                
                compressed_x = []
                index = min(max_compressed_block_size, len(dctn_x) - 1)
                while int(np.round(dctn_x[index] * scale)) == 0 and index > 0:
                    index = index - 1
                
                for i in range(1, index + 1):
                    compressed_x.append(int(np.round(dctn_x[i] * scale)))
                
                lossless_compressed_x = lossless_compress(compressed_x)
                result_x.append([lossless_compressed_x, mean])
            
            for block_id in range(0, (len(delta_y) - 1) // block_size + 1):
                block_y = delta_y[block_id * block_size: min((block_id + 1) * block_size, len(delta_y))]
                mean = np.mean(block_y)
                block_y = block_y - mean
                
                dctn_y = dctn(block_y)
                compressed_y = []
                index = min(max_compressed_block_size, len(dctn_y) - 1)
                while int(np.round(dctn_y[index] * scale)) == 0 and index > 0:
                    index = index - 1
                
                for i in range(1, index + 1):
                    compressed_y.append(int(np.round(dctn_y[i] * scale)))
                
                lossless_compressed_y = lossless_compress(compressed_y)
                result_y.append([lossless_compressed_y, mean])
        else:
            all_bits = ((2 * 64 * len(self.traj)) / compression_ratio - 10 * 32 - 2 * ((len(self.traj) - 2) // block_size + 1) * (8 + 64)) / 2
            
            for block_id in range(0, (len(delta_x) - 1) // block_size + 1):
                block_x = delta_x[block_id * block_size: min((block_id + 1) * block_size, len(delta_x))]
                mean = np.mean(block_x)
                block_x = block_x - mean
                
                dctn_x = dctn(block_x)
                compressed_x = []
                index = min(255, len(dctn_x) - 1)
                while int(np.round(dctn_x[index] * scale)) == 0 and index > 0:
                    index = index - 1
                
                for i in range(1, index + 1):
                    compressed_x.append(int(np.round(dctn_x[i] * scale)))
                
                lossless_compressed_x = lossless_compress(compressed_x, all_bits * len(block_x) / len(delta_x))
                result_x.append([lossless_compressed_x, mean])
            
            for block_id in range(0, (len(delta_y) - 1) // block_size + 1):
                block_y = delta_y[block_id * block_size: min((block_id + 1) * block_size, len(delta_y))]
                mean = np.mean(block_y)
                block_y = block_y - mean
                
                dctn_y = dctn(block_y)
                compressed_y = []
                index = min(255, len(dctn_y) - 1)
                while int(np.round(dctn_y[index] * scale)) == 0 and index > 0:
                    index = index - 1
                
                for i in range(1, index + 1):
                    compressed_y.append(int(np.round(dctn_y[i] * scale)))
                
                lossless_compressed_y = lossless_compress(compressed_y, all_bits * len(block_y) / len(delta_y))
                result_y.append([lossless_compressed_y, mean])
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(signint2bitarray(block_size, 32))
                write_bits.extend(signint2bitarray(scale, 32))
                
                write_bits.extend(double2bitarray(x_list[0]))
                write_bits.extend(signint2bitarray(len(delta_x), 32))
                write_bits.extend(signint2bitarray(len(result_x), 32))
                for i in range(len(result_x)):
                    write_bits.extend(signint2bitarray(len(result_x[i][0]), 8))
                    for j in range(len(result_x[i][0])):
                        write_bits.extend(signint2bitarray(result_x[i][0][j][0], 5))
                        write_bits.extend(signint2bitarray(len(result_x[i][0][j][1]), 8))
                        for k in range(len(result_x[i][0][j][1])):
                            write_bits.extend(signint2bitarray(result_x[i][0][j][1][k], result_x[i][0][j][0]))
                    write_bits.extend(double2bitarray(result_x[i][1]))
                
                write_bits.extend(double2bitarray(y_list[0]))
                write_bits.extend(signint2bitarray(len(delta_y), 32))
                write_bits.extend(signint2bitarray(len(result_y), 32))
                for i in range(len(result_y)):
                    write_bits.extend(signint2bitarray(len(result_y[i][0]), 8))
                    for j in range(len(result_y[i][0])):
                        write_bits.extend(signint2bitarray(result_y[i][0][j][0], 5))
                        write_bits.extend(signint2bitarray(len(result_y[i][0][j][1]), 8))
                        for k in range(len(result_y[i][0][j][1])):
                            write_bits.extend(signint2bitarray(result_y[i][0][j][1][k], result_y[i][0][j][0]))
                    write_bits.extend(double2bitarray(result_y[i][1]))
                
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            block_size = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            scale = 1
            scale = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            
            start_x = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            delta_x_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            x_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            result_x = []
            for i in range(x_num):
                x_block_num = bitarray2signint(read_bits[:8], False)
                read_bits = read_bits[8:]
                compressed_x = []
                for j in range(x_block_num):
                    bits = bitarray2signint(read_bits[:5], False)
                    read_bits = read_bits[5:]
                    temp_x_num = bitarray2signint(read_bits[:8], False)
                    read_bits = read_bits[8:]
                    for k in range(temp_x_num):
                        compressed_x.append(bitarray2signint(read_bits[:bits]))
                        read_bits = read_bits[bits:]
                mean = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                result_x.append([compressed_x, mean])
            
            start_y = bitarray2double(read_bits[:64])
            read_bits = read_bits[64:]
            delta_y_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            y_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            result_y = []
            for i in range(y_num):
                y_block_num = bitarray2signint(read_bits[:8], False)
                read_bits = read_bits[8:]
                compressed_y = []
                for j in range(y_block_num):
                    bits = bitarray2signint(read_bits[:5], False)
                    read_bits = read_bits[5:]
                    temp_y_num = bitarray2signint(read_bits[:8], False)
                    read_bits = read_bits[8:]
                    for k in range(temp_y_num):
                        compressed_y.append(bitarray2signint(read_bits[:bits]))
                        read_bits = read_bits[bits:]
                mean = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                result_y.append([compressed_y, mean])
        
        tot_idct_x = np.array([])
        for i in range(len(result_x)):
            compressed_x = result_x[i][0]
            mean = result_x[i][1]
            compressed_x = np.pad(compressed_x, (1, min(block_size, delta_x_num - block_size * i) - len(compressed_x) - 1), 'constant')
            idct_x = idctn(compressed_x / scale)
            idct_x = idct_x + mean
            tot_idct_x = np.concatenate((tot_idct_x, idct_x))
        
        tot_idct_y = np.array([])
        for i in range(len(result_y)):
            compressed_y = result_y[i][0]
            mean = result_y[i][1]
            compressed_y = np.pad(compressed_y, (1, min(block_size, delta_y_num - block_size * i) - len(compressed_y) - 1), 'constant')
            idct_y = idctn(compressed_y / scale)
            idct_y = idct_y + mean
            tot_idct_y = np.concatenate((tot_idct_y, idct_y))

        assert len(tot_idct_x) == len(tot_idct_y)
        
        decompress_traj = [np.array([start_x, start_y])]
        for i in range(len(tot_idct_x)):
            decompress_traj.append(decompress_traj[i] + np.array([tot_idct_x[i], tot_idct_y[i]]))
        self.decompress_traj = np.array(decompress_traj)

class CISEDSCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', delta_t=0.1):
        super().__init__('cised_s', path, data_source, delta_t)
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
                point = self.traj[index0] + (self.traj[index1] - self.traj[index0]) / (index1 - index0)
                point_list = []
                for j in range(polygon_num):
                    angle = (2 * j - 1) * np.pi / polygon_num
                    point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) / (index1 - index0))
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
                write_bits.extend(signint2bitarray(len(index_list), 32))
                for i in range(len(index_list)):
                    write_bits.extend(signint2bitarray(index_list[i], 32))
                    write_bits.extend(double2bitarray(self.traj[index_list[i]][0]))
                    write_bits.extend(double2bitarray(self.traj[index_list[i]][1]))
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            index_list = []
            point_list = []
            for i in range(length_num):
                index_list.append(bitarray2signint(read_bits[:32], False))
                read_bits = read_bits[32:]
                x = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                y = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                point_list.append([x, y])
        
        point_list = np.array(point_list)
        
        decompress_traj = []
        for i in range(len(index_list)):
            if i == len(index_list) - 1:
                decompress_traj.append(point_list[i])
            else:
                for j in range(index_list[i], index_list[i + 1]):
                    temp = (j - index_list[i]) / (index_list[i + 1] - index_list[i]) * (point_list[i + 1] - point_list[i]) + point_list[i]
                    decompress_traj.append(temp)
        
        self.decompress_traj = np.array(decompress_traj)
        
class CISEDWCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', delta_t=0.1):
        super().__init__('cised_w', path, data_source, delta_t)
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
                point = point0 + (point1 - point0) / (index1 - index0)
                point_list = []
                for j in range(polygon_num):
                    angle = (2 * j - 1) * np.pi / polygon_num
                    point_list.append(point + r * np.array([np.cos(angle), np.sin(angle)]) / (index1 - index0))
                return Polygon(point_list)
            
            temp_polygon = get_polygon(point_list[-1], self.traj[i], id_list[-1], i, max_error)
            if polygon.is_empty:
                polygon = temp_polygon
            else:
                center_point = np.array([polygon.centroid.x, polygon.centroid.y])
                new_polygon = polygon.intersection(temp_polygon)
                if new_polygon.is_empty:
                    add_point = self.traj[i - 1]
                    temp_point = point_list[-1] + (add_point - point_list[-1]) / (i - 1 - id_list[-1])
                    if not polygon.contains(Point(temp_point)):
                        add_point = point_list[-1] + (center_point - point_list[-1]) * (i - 1 - id_list[-1])
                    
                    point_list.append(add_point)
                    id_list.append(i - 1)
                    polygon = get_polygon(add_point, self.traj[i], id_list[-1], i, max_error)
                else:
                    polygon = new_polygon
        
        add_point = self.traj[-1]
        if not polygon.is_empty:
            center_point = np.array([polygon.centroid.x, polygon.centroid.y])
            temp_point = point_list[-1] + (add_point - point_list[-1]) / (len(self.traj) - 1 - id_list[-1])
            if not polygon.contains(Point(temp_point)):
                add_point = point_list[-1] + (center_point - point_list[-1]) * (len(self.traj) - 1 - id_list[-1])
        
        point_list.append(add_point)
        id_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(signint2bitarray(len(id_list), 32))
                for i in range(len(id_list)):
                    write_bits.extend(signint2bitarray(id_list[i], 32))
                    write_bits.extend(double2bitarray(point_list[i][0]))
                    write_bits.extend(double2bitarray(point_list[i][1]))
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            index_list = []
            point_list = []
            for i in range(length_num):
                index_list.append(bitarray2signint(read_bits[:32], False))
                read_bits = read_bits[32:]
                x = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                y = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                point_list.append([x, y])
        
        point_list = np.array(point_list)
        
        decompress_traj = []
        for i in range(len(index_list)):
            if i == len(index_list) - 1:
                decompress_traj.append(point_list[i])
            else:
                for j in range(index_list[i], index_list[i + 1]):
                    temp = (j - index_list[i]) / (index_list[i + 1] - index_list[i]) * (point_list[i + 1] - point_list[i]) + point_list[i]
                    decompress_traj.append(temp)
        
        self.decompress_traj = np.array(decompress_traj)

class DPCompressor(NoMapCompressor):
    def __init__(self, path, data_source='nuplan', delta_t=0.1):
        super().__init__('dp', path, data_source, delta_t)
        pass
    
    @profile
    def compress(self, save_path=None, save=True, **kwargs):
        if save_path is None:
            save_path = self.save_path
            
        max_error = kwargs['max_error'] if 'max_error' in kwargs else 0.3
        
        def divide(left, right):
            result = []
            max_dist = 0
            max_index = 0
            for i in range(left + 1, right):
                compressed_point = (i - left) / (right - left) * (self.traj[right] - self.traj[left]) + self.traj[left]
                dist = np.sqrt(np.sum((compressed_point - self.traj[i]) ** 2))
                if dist > max_dist:
                    max_dist = dist
                    max_index = i
            if max_dist > max_error:
                result.extend(divide(left, max_index))
                result.append(max_index)
                result.extend(divide(max_index, right))
            return result
        
        index_list = [0]
        index_list.extend(divide(0, len(self.traj) - 1))
        index_list.append(len(self.traj) - 1)
        
        if save:
            with open(save_path, 'wb') as f:
                write_bits = bitarray()
                write_bits.extend(signint2bitarray(len(index_list), 32))
                for i in range(len(index_list)):
                    write_bits.extend(signint2bitarray(index_list[i], 32))
                    write_bits.extend(double2bitarray(self.traj[index_list[i]][0]))
                    write_bits.extend(double2bitarray(self.traj[index_list[i]][1]))
                self.compress_bits = len(write_bits)
                write_bits.tofile(f)
        
    
    def decompress(self, load_path=None, **kwargs):
        if load_path is None:
            load_path = self.load_path
        
        with open(load_path, 'rb') as f:
            read_bits = bitarray()
            read_bits.fromfile(f)
            
            length_num = bitarray2signint(read_bits[:32], False)
            read_bits = read_bits[32:]
            index_list = []
            point_list = []
            for i in range(length_num):
                index_list.append(bitarray2signint(read_bits[:32], False))
                read_bits = read_bits[32:]
                x = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                y = bitarray2double(read_bits[:64])
                read_bits = read_bits[64:]
                point_list.append([x, y])
        
        point_list = np.array(point_list)
        
        decompress_traj = []
        for i in range(len(index_list)):
            if i == len(index_list) - 1:
                decompress_traj.append(point_list[i])
            else:
                for j in range(index_list[i], index_list[i + 1]):
                    temp = (j - index_list[i]) / (index_list[i + 1] - index_list[i]) * (point_list[i + 1] - point_list[i]) + point_list[i]
                    decompress_traj.append(temp)
        
        self.decompress_traj = np.array(decompress_traj)

def main():
    assert False, 'do not run this'
    
    # if not os.path.exists("cache/maps.pkl"):
    #     maps = {
    #             "sg-one-north" : NuplanMap(name = "sg-one-north"),
    #             "us-ma-boston" : NuplanMap(name = "us-ma-boston"),
    #             "us-nv-las-vegas-strip" : NuplanMap(name = "us-nv-las-vegas-strip"),
    #             "us-pa-pittsburgh-hazelwood" : NuplanMap(name = "us-pa-pittsburgh-hazelwood"),
    #             }
    #     with open("cache/maps.pkl", "wb") as f:
    #         pickle.dump(maps, f)
    # else:
    #     with open("cache/maps.pkl", "rb") as f:
    #         maps = pickle.load(f)
    
    path = "/nas/common/data/trajectory/nuplan/nuplan_csv/test"
    data_source = 'nuplan'
    # path = "/nas/common/data/trajectory/ShangQi-2023/saic_data/long_term"
    # data_source = 'shangqi'

    del_csvs = ['/nas/common/data/trajectory/ShangQi-2023/saic_data/long_term/eval_2023_06_05_09_37_28_']
    # if os.path.exists("cache/del_csvs"):
    #     with open("cache/del_csvs", "r") as f:
    #         data = f.readlines()
    #         del_csvs = [x.strip() for x in data]
    
    delta_t = 0.1
    
    with open(f'config/delta_t_{delta_t}.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    dct_no_map_block_size = config['dct_no_map']['block_size']
    dct_no_map_compression_ratio = config['dct_no_map']['compression_ratio']
    dct_no_map_scale = config['dct_no_map']['scale']
    cised_max_error = config['cised']['max_error']
    
    global_counter = Global_Counter()
    for dir_name in os.listdir(path):
        ff = os.path.join(path, dir_name)
        if ff in del_csvs or ff not in ['/nas/common/data/trajectory/nuplan/nuplan_csv/test/2021.09.22.01.45.32_veh-53_00298_00432', '/nas/common/data/trajectory/nuplan/nuplan_csv/test/2021.06.28.17.56.29_veh-47_01378_02853']:
            continue
        print(ff, flush=True)
        
        # origin_compressor = Compressor('', ff, maps)
        # print("origin performance:", origin_compressor.get_origin_metrics())
        
        # simple_compressor = SimpleCompressor(ff, maps)
        # simple_compressor.compress(bits=2)
        # simple_compressor.decompress()
        
        # print("simple performance:", simple_compressor.evaluation_metrics())
        
        # press_compressor = PressCompressor(ff, maps)
        # press_compressor.compress()
        # press_compressor.decompress()
        
        # print("press performance:", press_compressor.evaluation_metrics())
        
        # dct_compressor = DCTCompressor(ff, maps)
        # dct_compressor.compress()
        # dct_compressor.decompress()
        
        # print("dct performance:", dct_compressor.evaluation_metrics())
        
        dct_no_map_compressor = DCTNoMapCompressor(ff, data_source, delta_t)
        dct_no_map_compressor.compress(block_size=dct_no_map_block_size, compression_ratio=dct_no_map_compression_ratio, scale=dct_no_map_scale)
        dct_no_map_compressor.decompress()
        
        print("dct no map performance:", dct_no_map_compressor.evaluation_metrics())
        global_counter.count(dct_no_map_compressor.name, dct_no_map_compressor.evaluation_metrics())
        
        ciseds_compressor = CISEDSCompressor(ff, data_source, delta_t)
        ciseds_compressor.compress(max_error=cised_max_error)
        ciseds_compressor.decompress()
        
        print("cised s performance:", ciseds_compressor.evaluation_metrics())
        global_counter.count(ciseds_compressor.name, ciseds_compressor.evaluation_metrics())
        
        cisedw_compressor = CISEDWCompressor(ff, data_source, delta_t)
        cisedw_compressor.compress(max_error=cised_max_error)
        cisedw_compressor.decompress()
        
        print("cised w performance:", cisedw_compressor.evaluation_metrics())
        global_counter.count(cisedw_compressor.name, cisedw_compressor.evaluation_metrics())
        
        # temp_arg_max = dct_no_map_compressor.argmax_error
        # dct_no_map_compressor.plot_traj(temp_arg_max)
        # ciseds_compressor.plot_traj(temp_arg_max)
        # cisedw_compressor.plot_traj(temp_arg_max)
        
        # dp_compressor = DPCompressor(ff, data_source, delta_t)
        # dp_compressor.compress(max_error=cised_max_error)
        # dp_compressor.decompress()
        
        # print("dp performance:", dp_compressor.evaluation_metrics())
        # global_counter.count(dp_compressor.name, dp_compressor.evaluation_metrics())
        
    global_counter.get_result()
        
    # path = '/home/wkf/nas/PRESS/Geolife Trajectories 1.3'
    # for file_name in get_files(path):
    #     if not file_name.endswith('.plt'):
    #         continue
    #     print(file_name)
    #     dct_no_map_compressor = DCTNoMapCompressor(file_name, data_source='geolife')
    #     dct_no_map_compressor.compress()
    #     dct_no_map_compressor.decompress()
        
    #     print("dct no map performance:", dct_no_map_compressor.evaluation_metrics(), flush=True)
        
    
    
main()