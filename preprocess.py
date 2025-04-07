import os
from nuplan_map import *
import yaml
import pickle
import math

def get_data(line, title, name, default = np.float64(0)):
    if name in title:
        data = line[title.index(name)]
        if len(data):
            return np.float64(data)
        else:
            return default
    else:
        return default

def preprocess_nuplan_map():
    '''
    print error paths and skip them during compression
    '''
    path = "/nas/common/data/trajectory/nuplan/nuplan_csv/test"
    if not os.path.exists("cache/maps.pkl"):
        maps = {
                "sg-one-north" : NuplanMap(name = "sg-one-north"),
                "us-ma-boston" : NuplanMap(name = "us-ma-boston"),
                "us-nv-las-vegas-strip" : NuplanMap(name = "us-nv-las-vegas-strip"),
                "us-pa-pittsburgh-hazelwood" : NuplanMap(name = "us-pa-pittsburgh-hazelwood"),
                }
        with open("cache/maps.pkl", "wb") as f:
            pickle.dump(maps, f)
    else:
        with open("cache/maps.pkl", "rb") as f:
            maps = pickle.load(f)
    
    error_paths = []
    tot = 0
    for dir_name in os.listdir(path):
        tot = tot + 1
        ff = os.path.join(path, dir_name)
        
        if os.path.exists(os.path.join(ff, 'meta.yml')):
            with open(os.path.join(ff, 'meta.yml'), 'r') as file:
                config = yaml.safe_load(file)
        else:
            continue
        numplan_map = maps[config['map']]
        
        data = [x.strip().split(",") for x in open(os.path.join(ff, 'ego.csv')).readlines()]
        title = data[0]
        data = data[1:]
        traj = []
        for line in data:
            x = get_data(line, title, "X_utm(m)")
            y = get_data(line, title, "Y_utm(m)")
            traj.append([float(x), float(y)])
        
        traj = np.array(traj)
        result = numplan_map.get_data_from_trajectory(traj)
        if result is None:
            error_paths.append(ff)
            print(ff)

from pyproj import Proj, Transformer
import multiprocessing

def lonlat_to_utm(lon, lat, utm_zone=50):
    utm_proj = Proj(proj='utm', zone=utm_zone, south=False)
    wgs_proj = Proj(proj='latlong', datum='WGS84')
    
    transformer = Transformer.from_proj(wgs_proj, utm_proj, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return [x, y]

def geodetic_to_ecef(lon, lat, alt):
    a = 6378137.0
    e_sq = 0.00669437999014
    lambda_rad = math.radians(lon)
    phi_rad = math.radians(lat)
    
    sin_phi = math.sin(phi_rad)
    N = a / math.sqrt(1 - e_sq * sin_phi**2)
    
    x = (N + alt) * math.cos(phi_rad) * math.cos(lambda_rad)
    y = (N + alt) * math.cos(phi_rad) * math.sin(lambda_rad)
    z = (N * (1 - e_sq) + alt) * sin_phi
    
    return (x, y, z)
    
def process_file_geolife(ff, dir_name):
    all_data = []
    file_names = os.listdir(ff)
    file_names.sort()
    
    for file_name in file_names:
        temp_path = os.path.join(ff, file_name)
        data = [x.strip().split(",") for x in open(temp_path).readlines()]
        data = data[6:]
        all_data.extend(data)
        
    traj = []
    t_list = []
    for line in all_data:
        lat = np.float64(line[0])
        lon = np.float64(line[1])
        h = np.float64(line[3])
        t = np.float64(line[4]) * 86400
        
        if len(t_list) == 0 or abs(t - t_list[-1]) > 1e-6:
            t_list.append(t)
            data = lonlat_to_utm(lon, lat, utm_zone=35)
            assert not math.isinf(data[0]) and not math.isinf(data[1])
            traj.append(data)
    traj = np.array(traj)
            
    other_points = []
    index_list = []
    
    start_index = 0
    end_index = 0
    while start_index < len(t_list):
        end_index = start_index
        while end_index < len(t_list) - 1 and \
                t_list[end_index + 1] - t_list[end_index] < 60 and \
                  np.sqrt(np.sum((traj[end_index + 1] - traj[end_index]) ** 2) / (t_list[end_index + 1] - t_list[end_index])) < 100:
            end_index = end_index + 1
        
        if end_index - start_index <= 0:
            for i in range(start_index, end_index + 1):
                other_points.append(i)
            start_index = end_index + 1
            continue
        index_list.append([start_index, end_index])
        start_index = end_index + 1
    
    
    traj_path = os.path.join('/home/wkf/data/PRESS/datasets/geolife', f"traj_{dir_name}.txt")
    with open(traj_path, "w") as f:
        for j in range(len(traj)):
            f.write(f"{t_list[j]},{traj[j][0]},{traj[j][1]}\n")
    print(ff, flush=True)
    
    return len(other_points)
        
def preprocess_geolife():
    if os.path.exists('/home/wkf/data/PRESS/datasets/geolife'):
        os.system('rm -rf /home/wkf/data/PRESS/datasets/geolife')
        
    os.makedirs('/home/wkf/data/PRESS/datasets/geolife')
    
    path = '/home/wkf/data/Geolife Trajectories 1.3/Data'
    tot_path = []
    
    for dir_name in os.listdir(path):
        ff = os.path.join(path, dir_name)
        ff = os.path.join(ff, 'Trajectory')
        tot_path.append([ff, dir_name])
        
    with multiprocessing.Pool() as pool:
        results = []
        for ff, dir_name in tot_path:
            result = pool.apply_async(process_file_geolife, args=(ff, dir_name,))
            results.append(result)
        tot_cnt = 0
        for result in results:
            cnt = result.get()
            tot_cnt += cnt
    print(f"total other points: {tot_cnt}") # 649911
    
    print("preprocess geolife done")

def process_file_mopsi(ff, dir_name):
    all_data = []
    file_names = os.listdir(ff)
    file_names.sort()
    
    for file_name in file_names:
        temp_path = os.path.join(ff, file_name)
        data = [x.strip().split(" ") for x in open(temp_path).readlines()]
        all_data.extend(data)
        
    traj = []
    t_list = []
    for line in all_data:
        lat = np.float64(line[0])
        lon = np.float64(line[1])
        h = np.float64(line[3])
        t = np.float64(line[2]) / 1000
        
        if len(t_list) == 0 or abs(t - t_list[-1]) > 1e-6:
            assert len(t_list) == 0 or t > t_list[-1], t
            t_list.append(t)
            data = lonlat_to_utm(lon, lat, utm_zone=40)
            assert not math.isinf(data[0]) and not math.isinf(data[1])
            traj.append(data)
    traj = np.array(traj)
            
    other_points = []
    index_list = []
    
    start_index = 0
    end_index = 0
    while start_index < len(t_list):
        end_index = start_index
        while end_index < len(t_list) - 1 and \
                t_list[end_index + 1] - t_list[end_index] < 60 and \
                  np.sqrt(np.sum((traj[end_index + 1] - traj[end_index]) ** 2) / (t_list[end_index + 1] - t_list[end_index])) < 100:
            end_index = end_index + 1
        
        if end_index - start_index <= 10:
            for i in range(start_index, end_index + 1):
                other_points.append(i)
            start_index = end_index + 1
            continue
        index_list.append([start_index, end_index])
        start_index = end_index + 1
    
    
    traj_path = os.path.join('/home/wkf/data/PRESS/datasets/mopsi', f"traj_{dir_name}.txt")
    with open(traj_path, "w") as f:
        for j in range(len(traj)):
            f.write(f"{t_list[j]},{traj[j][0]},{traj[j][1]}\n")
    print(ff, flush=True)
    
    return len(other_points)
    
def preprocess_mopsi():
    if os.path.exists('/home/wkf/data/PRESS/datasets/mopsi'):
        os.system('rm -rf /home/wkf/data/PRESS/datasets/mopsi')
        
    os.makedirs('/home/wkf/data/PRESS/datasets/mopsi')
    
    path = '/home/wkf/data/mopsi_routes'
    tot_path = []
    
    for dir_name in os.listdir(path):
        ff = os.path.join(path, dir_name)
        tot_path.append([ff, dir_name])
        
    with multiprocessing.Pool() as pool:
        results = []
        for ff, dir_name in tot_path:
            result = pool.apply_async(process_file_mopsi, args=(ff, dir_name,))
            results.append(result)
        tot_cnt = 0
        for result in results:
            cnt = result.get()
            tot_cnt += cnt
    print(f"total other points: {tot_cnt}")
    
    print("preprocess mopsi done")
    
def process_file_uav(ff, dir_name):
    all_data = [x.strip().split(",") for x in open(ff).readlines()][2:]
    
    traj_dict = {}
    t_dict = {}
    
    for i in range(len(all_data)):
        t = np.float64(all_data[i][0])
        index = int(all_data[i][1])
        lat = np.float64(all_data[i][3])
        lon = np.float64(all_data[i][4])
        alt = np.float64(all_data[i][5])
        
        if index not in traj_dict:
            traj_dict[index] = []
            t_dict[index] = []
            
        data = lonlat_to_utm(lon, lat, utm_zone=31)
        assert not math.isinf(data[0]) and not math.isinf(data[1])
        
        if len(t_dict[index]) == 0 or abs(t - t_dict[index][-1]) > 1e-6:
            traj_dict[index].append([data[0], data[1], alt])
            t_dict[index].append(t)
    
    for index in traj_dict.keys():
        traj = traj_dict[index]
        t_list = t_dict[index]
        traj_path = os.path.join('/home/wkf/data/PRESS/datasets/uav', f"traj_{dir_name}_{index}.txt")
        with open(traj_path, "w") as f:
            for j in range(len(traj)):
                f.write(f"{t_list[j]},{traj[j][0]},{traj[j][1]},{traj[j][2]}\n")
    print(ff, flush=True)
    
def preprocess_uav():
    if os.path.exists('/home/wkf/data/PRESS/datasets/uav'):
        os.system('rm -rf /home/wkf/data/PRESS/datasets/uav')
        
    os.makedirs('/home/wkf/data/PRESS/datasets/uav')
    
    path = '/home/wkf/data/uavdelievery'
    tot_path = []
    
    for dir_name in os.listdir(path):
        ff = os.path.join(path, dir_name)
        tot_path.append([ff, dir_name])
        
    with multiprocessing.Pool() as pool:
        results = []
        for ff, dir_name in tot_path:
            result = pool.apply_async(process_file_uav, args=(ff, dir_name,))
            results.append(result)
        tot_cnt = 0
        for result in results:
            result.get()
    print(f"total other points: {tot_cnt}")
    
    print("preprocess uav done")
    
def process_file_geolife_3d(ff, dir_name):
    all_data = []
    file_names = os.listdir(ff)
    file_names.sort()
    
    for file_name in file_names:
        temp_path = os.path.join(ff, file_name)
        data = [x.strip().split(",") for x in open(temp_path).readlines()]
        data = data[6:]
        all_data.extend(data)
        
    traj = []
    t_list = []
    for line in all_data:
        lat = np.float64(line[0])
        lon = np.float64(line[1])
        h = np.float64(line[3]) * 0.3048
        t = np.float64(line[4]) * 86400
        
        if len(t_list) == 0 or abs(t - t_list[-1]) > 1e-6:
            t_list.append(t)
            data = geodetic_to_ecef(lon, lat, h)
            traj.append(data)
    traj = np.array(traj)
            
    other_points = []
    index_list = []
    
    start_index = 0
    end_index = 0
    while start_index < len(t_list):
        end_index = start_index
        while end_index < len(t_list) - 1 and \
                t_list[end_index + 1] - t_list[end_index] < 60 and \
                  np.sqrt(np.sum((traj[end_index + 1] - traj[end_index]) ** 2) / (t_list[end_index + 1] - t_list[end_index])) < 100:
            end_index = end_index + 1
        
        if end_index - start_index <= 10:
            for i in range(start_index, end_index + 1):
                other_points.append(i)
            start_index = end_index + 1
            continue
        index_list.append([start_index, end_index])
        start_index = end_index + 1
    
    
    traj_path = os.path.join('/home/wkf/data/PRESS/datasets/geolife_3d', f"traj_{dir_name}.txt")
    with open(traj_path, "w") as f:
        for j in range(len(traj)):
            f.write(f"{t_list[j]},{traj[j][0]},{traj[j][1]},{traj[j][2]}\n")
    print(ff, flush=True)
    
    return len(other_points)
    
def preprocess_geolife_3d():
    if os.path.exists('/home/wkf/data/PRESS/datasets/geolife_3d'):
        os.system('rm -rf /home/wkf/data/PRESS/datasets/geolife_3d')
        
    os.makedirs('/home/wkf/data/PRESS/datasets/geolife_3d')
    
    path = '/home/wkf/data/Geolife Trajectories 1.3/Data'
    tot_path = []
    
    for dir_name in os.listdir(path):
        ff = os.path.join(path, dir_name)
        ff = os.path.join(ff, 'Trajectory')
        tot_path.append([ff, dir_name])
        
    with multiprocessing.Pool() as pool:
        results = []
        for ff, dir_name in tot_path:
            result = pool.apply_async(process_file_geolife_3d, args=(ff, dir_name,))
            results.append(result)
        tot_cnt = 0
        for result in results:
            cnt = result.get()
            tot_cnt += cnt
    print(f"total other points: {tot_cnt}")
    
    print("preprocess geolife_3d done")

preprocess_geolife()