from no_map_compress import *

import multiprocessing


def count_file(path):
    data = [x.strip().split(",") for x in open(path).readlines()]
    
    traj = []
    t_list = []
    
    for line in data:
        t = np.float64(line[0])
        x = np.float64(line[1])
        y = np.float64(line[2])
        t_list.append(t)
        traj.append([x, y])
        
    
    traj = np.array(traj)
            
    other_points = []
    index_list = []
    
    start_index = 0
    end_index = 0
    while start_index < len(t_list):
        end_index = start_index
        while end_index < len(t_list) - 1 and \
                t_list[end_index + 1] - t_list[end_index] < 60 and \
                  np.sqrt(np.sum((traj[end_index + 1] - traj[end_index]) ** 2) / (t_list[end_index + 1] - t_list[end_index])) < 1000:
            end_index = end_index + 1
        
        if end_index - start_index <= 0:
            for i in range(start_index, end_index + 1):
                other_points.append(i)
            start_index = end_index + 1
            continue
        index_list.append([start_index, end_index])
        start_index = end_index + 1
        
    return len(other_points)

def count_files():
    path = '/home/wkf/data/PRESS/datasets/geolife'
    tot_path = []

    for file_name in os.listdir(path):
        ff = os.path.join(path, file_name)
        tot_path.append(ff)
        
    with multiprocessing.Pool(processes=16) as pool:
        pool_results = []
        for ff in tot_path:
            result = pool.apply_async(count_file, args=(ff,))
            pool_results.append(result)
        
        count = 0
        for result in pool_results:
            res = result.get()
            count += res
        print(count)
        
count_files()