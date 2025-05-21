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
    
    # data = [x.strip().split(",") for x in open(os.path.join(path, 'ego.csv')).readlines()]
    # title = data[0]
    # data = data[1:]
    
    # def get_data(line, title, name, default = np.float64(0)):
    #     if name in title:
    #         data = line[title.index(name)]
    #         if len(data):
    #             return np.float64(data)
    #         else:
    #             return default
    #     else:
    #         return default
    
    # traj = []
    # t_list = []
    # for line in data:
    #     x = get_data(line, title, "X_utm(m)")
    #     y = get_data(line, title, "Y_utm(m)")
    #     t = get_data(line, title, "SimTime(s)")
    #     t_list.append(t)
    #     traj.append([x, y])
    
    # traj = np.array(traj)
    # t_list = np.array(t_list)
            
    other_points = []
    index_list = []
    
    start_index = 0
    end_index = 0
    
    tot_dist = 0
    tot_time = 0
    tot_cnt = 0
    
    test_cnt = 0
    
    while start_index < len(t_list):
        end_index = start_index
        while end_index < len(t_list) - 1 and \
                t_list[end_index + 1] - t_list[end_index] < 120 and \
                  np.sqrt(np.sum((traj[end_index + 1] - traj[end_index]) ** 2) / (t_list[end_index + 1] - t_list[end_index])) < 200:
            end_index = end_index + 1
        
        if end_index - start_index <= 1:
            for i in range(start_index, end_index + 1):
                if len(other_points) == 0 or i - other_points[-1] > 1:
                    test_cnt += 1
                
                other_points.append(i)
                
            start_index = end_index + 1
            continue
        index_list.append([start_index, end_index])
        
        temp_traj = traj[start_index:end_index + 1]
        temp_t_list = t_list[start_index:end_index + 1]
        
        temp_dist = np.sqrt(np.sum((temp_traj[1:] - temp_traj[:-1]) ** 2, axis=1))
        tot_dist += np.sum(temp_dist)
        temp_time = temp_t_list[-1] - temp_t_list[0]
        tot_time += temp_time
        tot_cnt += end_index - start_index
        
        
        start_index = end_index + 1
        
    return tot_dist, tot_time, tot_cnt, len(other_points), test_cnt

def count_files():
    tot_path = []
    path = '/home/wkf/data/PRESS/datasets/geolife' # 9.157710830943056 2.9696855649856775
    # path = '/home/wkf/data/PRESS/datasets/mopsi' # 5.477687821243067 2.115325000366175
    # path = '/nas/common/data/trajectory/nuplan/nuplan_csv/test' # 4.375608316533668 0.09999930041108658

    for file_name in os.listdir(path):
        ff = os.path.join(path, file_name)
        tot_path.append(ff)
        
    with multiprocessing.Pool(processes=16) as pool:
        pool_results = []
        for ff in tot_path:
            result = pool.apply_async(count_file, args=(ff,))
            pool_results.append(result)
        
        dist = 0
        time = 0
        cnt = 0
        
        test_len = 0
        test_cnt = 0
        
        for result in pool_results:
            res = result.get()
            dist += res[0]
            time += res[1]
            cnt += res[2]
            test_len += res[3]
            test_cnt += res[4]
        print(dist, time, cnt, dist / time, time / cnt)
        print(test_len, test_cnt, test_len / test_cnt)
        
count_files()