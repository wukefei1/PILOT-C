
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
from scipy.fft import *
import matplotlib.pyplot as plt
import matplotlib as mpl

# 输入信号
t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
x = np.array([0, 2, 4, 6, 8, 10, 10, 10, 10, 10, 8, 6, 4, 2, 0])
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
sumx = np.cumsum(x)
sumx = np.append(np.float64(0), sumx)
sumy = np.cumsum(y)
sumy = np.append(np.float64(0), sumy)

# N = len(x)
dctn_x = dctn(x, norm='ortho')
dctn_y = dctn(y, norm='ortho')

mpl.rcParams['font.size'] = 12
# plt.xlim(0, 15)

# plt.plot(t, x)
# plt.xlabel("T")
# plt.ylabel("V")
# plt.savefig("images/velocity_time_curve.png")

# plt.hlines(y=0, xmin=0, xmax=15)
# plt.plot(t, dctn_x)
# plt.xlabel("T")
# plt.ylabel("dctV")
# plt.savefig("images/DCT_result.png")


# plt.plot(t, x)
# plt.xlabel("T")
# plt.ylabel("VX")
# plt.savefig("images/VX.png")

# plt.plot(t, y)
# plt.xlabel("T")
# plt.ylabel("VY")
# plt.savefig("images/VY.png")

# plt.hlines(y=0, xmin=0, xmax=15)
# plt.plot(t, dctn_x)
# plt.xlabel("T")
# plt.ylabel("dctX")
# plt.savefig("images/dctX.png")

# plt.plot(t, dctn_y)
# plt.xlabel("T")
# plt.ylabel("dctY")
# plt.savefig("images/dctY.png")

plt.ylim(0, 15)
# plt.plot(sumx, t, label='original')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("images/trajectory.png")

# print(dctn_x, dctn_y)
# print(np.round(dctn_x / 10))
# print(np.round(dctn_y / 10))

# round_dctn_x = np.array([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# round_dctn_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# idctx = idctn(round_dctn_x * 10, norm='ortho') + np.mean(x)
# idcty = idctn(round_dctn_y * 10, norm='ortho') + np.mean(y)

# sum_idctx = np.cumsum(idctx)
# sum_idctx = np.append(np.float64(0), sum_idctx)
# sum_idcty = np.cumsum(idcty)
# sum_idcty = np.append(np.float64(0), sum_idcty)

# print(sum_idctx, sum_idcty)
# print(sum_idctx - sumx, sum_idcty - sumy)

# # plt.plot(sum_idctx, sum_idcty, label="reconstructed")
# # plt.xlabel("X")
# # plt.ylabel("Y")

# # plt.legend()

# # plt.savefig("images/trajectory_remove_error_points.png")

# # print(utfint2bitarray(10, 4, False))
# # print(utfint2bitarray(-100, 4, True))

# print(bitarray2utfint(utfint2bitarray(0, 6, True), 0, 6, True))

from no_map_compress import *

x = np.array([200, -200])

y = np.ones(400)

y[200:] = -1

dctn_x = np.round(dctn(x, norm='ortho'))
dctn_y = np.round(dctn(y, norm='ortho'))

print(dctn_x, dctn_y)

print(lossless_compress(dctn_x, 2, 4, bits=None))
print(lossless_compress(dctn_y[0: 30], 5, 4, bits=None))

idctx = idctn(dctn_x, norm='ortho')
idcty = idctn(dctn_y, norm='ortho')


sum_idctx = np.cumsum(idctx)
sum_idctx = np.append(np.float64(0), sum_idctx)
sum_idcty = np.cumsum(idcty)
sum_idcty = np.append(np.float64(0), sum_idcty)
print(sum_idctx, sum_idcty[200])