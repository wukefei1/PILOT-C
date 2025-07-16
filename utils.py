import os
from bitarray import bitarray
import struct

def is_leaf_directory(path):
    for root, dirs, files in os.walk(path):
        if dirs:
            return False
    return True

def get_leaf_directories(root_dir):
    leaf_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if is_leaf_directory(dir_path):
                leaf_dirs.append(dir_path)
    return leaf_dirs

def get_files(root_dir):
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            all_files.append(file_path)
    return all_files

def signint2bitarray(value, bit_length):
    if value < 0:
        value = (1 << bit_length) + value
    bit_array = bitarray()
    for i in range(bit_length - 1, -1, -1):
        bit_array.append((value >> i) & 1)
    return bit_array

def bitarray2signint(bit_array, signed=True):
    bit_length = len(bit_array)
    if bit_length == 0:
        return 0
    value = 0
    for i in range(bit_length):
        bit = bit_array[bit_length - 1 - i]
        value |= (bit << i)
    if signed and bit_array[0]:
        value = -(~value & ((1 << bit_length) - 1)) - 1
    return value

def customint2bitarray(x):
    if x == 0:
        return bitarray('0')
    if x == 1:
        return bitarray('100')
    if x == -1:
        return bitarray('101')
    
    sign = '0' if x > 0 else '1'
    abs_x = abs(x)
    k = abs_x.bit_length()  # 确定层级
    prefix = '1' * k + '0'
    value_part = abs_x - (1 << (k - 1))
    value_bits = bin(value_part)[2:].zfill(k - 1)
    return bitarray(prefix + sign + value_bits)

def bitarry2customint(bit_array, read_index):
    if bit_array[read_index] == 0:
        return 0, read_index + 1
    k = 0
    while bit_array[read_index] == 1:
        k += 1
        read_index += 1
    read_index += 1
    sign = bit_array[read_index]
    read_index += 1
    value_bits = ''.join(str(bit) for bit in bit_array[read_index:read_index+k - 1])
    read_index += k - 1
    
    value = (1 << (k - 1)) + (int(value_bits, 2) if value_bits != '' else 0)
    if sign == 1:
        value = -value
    return value, read_index

def double2bitarray(n):
    packed_double = struct.pack('d', n)
    bit_array = bitarray()
    for byte in packed_double:
        for i in range(8):
            bit_array.append((byte >> i) & 1)
    return bit_array

def bitarray2double(bit_array):
    packed_double = bytearray(8)
    for i in range(64):
        byte_index = i // 8
        bit_index = i % 8
        if bit_array[i]:
            packed_double[byte_index] |= (1 << bit_index)
    n = struct.unpack('d', packed_double)
    return n[0]

def float2bitarray(n):
    packed_float = struct.pack('f', n)
    bit_array = bitarray()
    for byte in packed_float:
        for i in range(8):
            bit_array.extend([(byte >> i) & 1])
    return bit_array

def bitarray2float(bit_array):
    packed_float = bytearray(4)
    for i in range(32):
        byte_index = i // 8
        bit_index = i % 8
        if bit_array[i]:
            packed_float[byte_index] |= (1 << bit_index)
    n = struct.unpack('f', packed_float)
    return n[0]

def varint2bitarray(value, data_length = 4):
    blocks = []
    while True:
        data = value & ((1 << data_length) - 1)
        value >>= data_length
        has_more = value > 0
        flag = 1 if has_more else 0
        blocks.append((flag, data))
        if not has_more:
            break
    
    bit_str = ''.join(
        str(flag) + bin(data)[2:].zfill(data_length)
        for flag, data in blocks
    )
    
    return bitarray(bit_str)

def bitarray2varint(bit_array, read_index, data_length = 4):
    flag = 1
    blocks = []
    while read_index < len(bit_array) and flag == 1:
        chunk = bit_array[read_index:read_index + data_length + 1]
        flag = chunk[0]
        data_bits = chunk[1:]
        data_bits = ''.join(str(bit) for bit in data_bits)
        data = int(data_bits, 2)
        blocks.append((flag, data))
        read_index += data_length + 1
    
    data_blocks = []
    for flag, data in blocks:
        data_blocks.append(data)
    
    result = 0
    for i, data in enumerate(data_blocks):
        result |= data << (data_length * i)
    
    return result, read_index

def zigzag2bitarray(value, data_length = 4):
    if value < 0:
        value = -2 * value - 1
    else:
        value = 2 * value
    return varint2bitarray(value, data_length)

def bitarray2zigzag(bit_array, read_index, data_length = 4):
    value, read_index = bitarray2varint(bit_array, read_index, data_length)
    if value & 1 == 0:
        return value >> 1, read_index
    else:
        return -((value + 1) >> 1), read_index

def count_bits(value: int):
    import math
    needbits = 0
    if value > 0:
        needbits = 1 + math.ceil(math.log2(value + 1))
    if value < 0:
        needbits = 1 + math.ceil(math.log2(-value))
    return needbits
    

class Global_Counter:
    def __init__(self):
        self.counter = {}
        
    def count(self, key, data):
        if key not in self.counter:
            self.counter[key] = {'max_error': 0,
                               'mean_error': 0,
                               'compress_ratio': 0,
                               'total_length': 0,
                               'total_count': 0,
                               'total_error_cnt': 0}
        self.counter[key]['mean_error'] = (self.counter[key]['mean_error'] * (self.counter[key]['total_length']) + data['mean_error'] * data['length']) / (self.counter[key]['total_length'] + data['length'])
        self.counter[key]['max_error'] = max(self.counter[key]['max_error'], data['max_error'])
        self.counter[key]['compress_ratio'] = (self.counter[key]['compress_ratio'] * (self.counter[key]['total_length']) + data['compress_ratio'] * data['length']) / (self.counter[key]['total_length'] + data['length'])
        self.counter[key]['total_length'] += data['length']
        self.counter[key]['total_count'] += 1
        self.counter[key]['total_error_cnt'] += data['error_cnt']
        
    def get_result(self):
        result = ''
        for key in self.counter:
            result += (f'{key}: {self.counter[key]}\n')
        return result