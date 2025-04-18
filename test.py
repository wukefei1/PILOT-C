
import struct
import numpy as np
import wave
import struct
import math
from utils import *

bits_per_sample = 16
max_sample_value = 2 ** (bits_per_sample - 1) - 1

def generate_audio(data, output_file="output.wav", ratio=None):
    sample_rate = 44100
    channels = 1

    if ratio is None:
        max_data = np.max(np.abs(np.array(data)))
        ratio = max_sample_value / max_data if max_data > 0 else 0

    print(len(data))
    for i in range(len(data)):
        data[i] = int(data[i] * ratio)

    with wave.open(output_file, 'wb') as f:
        f.setnchannels(channels)
        f.setsampwidth(bits_per_sample // 8)
        f.setframerate(sample_rate)
        f.writeframes(b''.join(struct.pack('h', int(x)) for x in data))
    return ratio

def compress_audio(input_file, output_file="output.mp3"):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(input_file, format="wav")
    compressed_audio = audio.set_frame_rate(44100).set_channels(1)
    compressed_audio.export(output_file, format="mp3")

def get_audio_samples(input_file):
    from pydub import AudioSegment
    from pydub.utils import make_chunks
    
    audio = AudioSegment.from_file(input_file, input_file.split(".")[-1])
    
    chunk_length_ms = 1000
    
    data = []
    for chunk in make_chunks(audio, chunk_length_ms):
        for sample in chunk.get_array_of_samples():
            data.append(sample)
    
    print(len(data))
    return data

def write_trajectory_to_bin(file_path, trajectory):
    with open(file_path, 'wb') as file:
        for x, y in trajectory:
            file.write(struct.pack('dd', x, y))

def read_trajectory_from_bin(file_path):
    with open(file_path, 'rb') as file:
        trajectory = []
        while True:
            x = file.read(8)
            y = file.read(8)
            if not x or not y:
                break
            trajectory.append([struct.unpack('d', x)[0], struct.unpack('d', y)[0]])
        return trajectory

def trajectory_compression(csv_path, output_path):
    data = [x.strip().split(",") for x in open(csv_path).readlines()]
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
    
    # write_trajectory_to_bin("origin.bin", traj)
    traj = np.array(traj)
    traj_x = traj[:, 0].copy()
    traj_y = traj[:, 1].copy()
    t = len(traj)
    
    x0 = traj_x[0]
    deltax = np.linspace(0, traj_x[-1] - x0, t)
    traj_x -= x0
    traj_x -= deltax
    traj_x = np.pad(traj_x, (0, (44 - (len(traj_x) % 44))), 'constant', constant_values=(0, 0))
    
    ratio = generate_audio(traj_x, "x.wav")
    compress_audio("x.wav", "x.mp3")
    compressed_x = get_audio_samples("x.mp3")
    caculated_x = []
    max_diff = 0
    for i in range(t):
        caculated_x.append(x0 + deltax[i] + compressed_x[i] / ratio)
        max_diff = max(max_diff, abs(caculated_x[-1] - traj[i][0]))
    print("max diff:", max_diff)
    
        
# # trajectory_compression("ego.csv", ".")
import numpy as np
from scipy.fft import dct, dctn, idctn

# 输入信号
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# x = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5])
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

x = x - np.mean(x)
N = len(x)

# 默认情况（无归一化）
X_default = dct(x, norm=None)
# print("Default DCT (no normalization):\n", X_default)

# 正交归一化
X_ortho = dct(x, norm='ortho')
# print("\nOrthonormalized DCT:\n", X_ortho)

# 手动计算正交归一化结果
manual_ortho = np.zeros_like(X_ortho)
manual_ortho[0] = np.sqrt(1/N) * np.sum(x)
for k in range(1, N):
    manual_ortho[k] = np.sqrt(2/N) * np.sum(x * np.cos(np.pi * k * (2*np.arange(N) + 1) / (2*N)))
# print("\nManual Orthonormalized DCT:\n", manual_ortho)

print(dctn(x))

idct_x = idctn(np.pad(dctn(x)[:12], (0, 24 - 12), 'constant'))

sum_x = np.cumsum(x)
sum_idct_x = np.cumsum(idct_x)
print(np.sum(manual_ortho ** 2 * 2 * N))
print(np.sum(X_ortho ** 2 * 2 * N))
print(np.sum(X_default ** 2))


print(bitarray2float(float2bitarray(0.6)))