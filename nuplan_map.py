from grid_tree import GridTree

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

import shapely
from shapely import Point
import numpy as np
from numpy import array
import codecs
import csv
import json
from tqdm import tqdm
from math import floor
from typing import Literal

DEBUG = False

class NuplanMap():
    # map_name = "us-nv-las-vegas-strip"
    DEFAULT_NAME = "sg-one-north"
    NUPLAN_MAP_VERSION = r"nuplan-maps-v1.0"
    NUPLAN_MAPS_ROOT = r"/nas/common/data/trajectory/nuplan/nuplan_map/maps"
    NAME_TO_EPSG = {"sg-one-north": "32648",
                    "us-ma-boston": "32619",
                    "us-nv-las-vegas-strip": "32611",
                    "us-pa-pittsburgh-hazelwood": "32617"}
    
    def __init__(self, version = NUPLAN_MAP_VERSION, root = NUPLAN_MAPS_ROOT, name=DEFAULT_NAME):
        import warnings
        warnings.filterwarnings("ignore")
        self.map_name = name
        map_db = GPKGMapsDB(version, root)
        self.map_api = NuPlanMap(map_db, map_name=self.map_name)
        
        self.boundary = Point([0,0]).buffer(100000000)
        # self.boundary = Point([664400, 3997000]).buffer(50)
        
        self.lanecenterlines = {}
        self.lanecenterlineconnectors = {}
        self.laneboundaries = {}
        self.laneboundaryconnectors = {}
        self.intersections = {}
        self.R2LC = {}
        self.RC2LCC = {}
        self.LCCstatus = {}
        
        self.point_geometry = np.zeros((0, 2))
        self.point_data = []
        
        self._get_intersections()
        self._get_real_lanes()
        self._get_virtual_lanes()
        self.grid_tree = GridTree(self.point_geometry, np.arange(len(self.point_geometry)))
        print(name + " init complete")
        
        # import pdb
        # pdb.set_trace()
        # print(self.lanecenterlines)
        # print(self.laneboundaries)
        # print(self.lanecenterlineconnectors)
        # print(self.laneboundaryconnectors)
        # print(self.intersections)
        # print(self.R2LC)
        # print(self.RC2LCC)
        
        
        # from geopandas import GeoSeries
        # import matplotlib.pyplot as plt

        # geometries = []
        # for lane in self.lanecenterlines.values():
        #     geometries.append(shapely.LineString(lane["geometry"]))
        # for lane in self.lanecenterlineconnectors.values():
        #     geometries.append(shapely.LineString(lane["geometry"]))
        # p = GeoSeries(geometries)
        # p.plot()
        # plt.show()
        # plt.savefig("test.png")
    
    def get_data_from_location(self, x, y, z=None):
        _, min_label = self.grid_tree.query(array([x, y]))
        point = self.point_data[min_label]
        
        if point["is_virtual"]:
            return self._get_data_from_location_and_LCC_id(x, y, point["L_id"])
        else:
            return self._get_data_from_location_and_LC_id(x, y, point["L_id"])
        
    def lane_is_in_intersection(self, L_id):
        return (L_id in self.lanecenterlineconnectors.keys())
    
    def lane_has_traffic_control_measure(self, L_id):
        if L_id in self.lanecenterlineconnectors.keys():
            return self.lanecenterlineconnectors[L_id]["has_traffic_light"]
        return False
    
    def get_lane_segment_centerline(self, L_id):
        if L_id in self.lanecenterlineconnectors.keys():
            return self.lanecenterlineconnectors[L_id]["geometry"]
        else:
            return self.lanecenterlines[L_id]["geometry"]
    
    def get_lane_ids_in_xy_bbox(self, x, y, r):
        temp_boundary = Point([x,y]).buffer(r)
        raw_lanes = self.map_api._get_proximity_map_object(temp_boundary, SemanticMapLayer.LANE)
        raw_lane_connectors = self.map_api._get_proximity_map_object(temp_boundary, SemanticMapLayer.LANE_CONNECTOR)
        ids = []
        for raw_lane in raw_lanes:
            ids.append(int(raw_lane.id))
        for raw_lane_connector in raw_lane_connectors:
            ids.append(int(raw_lane_connector.id))
        return ids
    
    # def get_lanecenterlines_dist(self, LC_id):
    #     temp_lcs0 = self.lanecenterlines[LC_id]["geometry"][:-1]
    #     temp_lcs1 = self.lanecenterlines[LC_id]["geometry"][1:]
    #     dist = np.sqrt(np.sum((temp_lcs0 - temp_lcs1) ** 2, 1))
    #     return sum(dist)
    
    # def get_lanecenterlineconnectors_dist(self, LCC_id):
    #     temp_lcc0 = self.lanecenterlineconnectors[LCC_id]["geometry"][:-1]
    #     temp_lcc1 = self.lanecenterlineconnectors[LCC_id]["geometry"][1:]
    #     dist = np.sqrt(np.sum((temp_lcc0 - temp_lcc1) ** 2, 1))
    #     return sum(dist)
    
    def _get_data_from_location_and_R_id(self, x, y, R_id):
        res = None
        for LC_id in self.R2LC[R_id]:
            temp = self._get_data_from_location_and_LC_id(x, y, LC_id)
            if res is None or abs(temp["dist"]) < abs(res["dist"]):
                res = temp
        return res
    
    def _get_data_from_location_and_RC_id(self, x, y, RC_id):
        res = None
        for LCC_id in self.RC2LCC[RC_id]:
            temp = self._get_data_from_location_and_LCC_id(x, y, LCC_id)
            if res is None or abs(temp["dist"]) < abs(res["dist"]):
                res = temp
        return res
    
    def _get_data_from_location_and_LC_id(self, x, y, LC_id):
        temp = np.sum((self.lanecenterlines[LC_id]["geometry"] - array([[x, y]] * self.lanecenterlines[LC_id]["geometry"].shape[0])) ** 2, axis=1) ** 0.5
        lc_point_id = np.argmin(temp)
        
        temp_lcs0 = self.lanecenterlines[LC_id]["geometry"][:-1]
        temp_lcs1 = self.lanecenterlines[LC_id]["geometry"][1:]
        dist = np.sqrt(np.sum((temp_lcs0 - temp_lcs1) ** 2, 1))
        dist = np.cumsum(dist)
        dist = np.concatenate(([0], dist))
        
        point0 = [x, y]
        
        if lc_point_id == 0 or (lc_point_id != self.lanecenterlines[LC_id]["geometry"].shape[0] - 1 and 
                                (sum((self.lanecenterlines[LC_id]["geometry"][lc_point_id + 1] - array([x, y])) ** 2, 1) ** 0.5 <
                                sum((self.lanecenterlines[LC_id]["geometry"][lc_point_id - 1] - array([x, y])) ** 2, 1) ** 0.5)):
            point1 = self.lanecenterlines[LC_id]["geometry"][lc_point_id]
            point2 = self.lanecenterlines[LC_id]["geometry"][lc_point_id + 1]
            length = dist[lc_point_id]
        else:
            point1 = self.lanecenterlines[LC_id]["geometry"][lc_point_id - 1]
            point2 = self.lanecenterlines[LC_id]["geometry"][lc_point_id]
            length = dist[lc_point_id - 1]
            
        v1 = array(point0) - array(point1)
        v2 = array(point2) - array(point1)
        s = v1[0] * v2[1] - v1[1] * v2[0]
        h = s / (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        length = length + np.dot(v1, v2) / np.dot(v2, v2) * (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        
        res = {}
        res["is_virtual"] = False
        res["LC_id"] = LC_id
        res["dist"] = h
        res["len"] = length
        res["R_id"] = self.lanecenterlines[LC_id]["R_id"]
        res["in_Section_id"] = self.lanecenterlines[LC_id]["in_Section_id"]
        res["out_Section_id"] = self.lanecenterlines[LC_id]["out_Section_id"]
        res["X"] = x
        res["Y"] = y
        return res
    
    def _get_data_from_location_and_LCC_id(self, x, y, LCC_id):
        temp = np.sum((self.lanecenterlineconnectors[LCC_id]["geometry"] - array([[x, y]] * self.lanecenterlineconnectors[LCC_id]["geometry"].shape[0])) ** 2, axis=1) ** 0.5
        lcc_point_id = np.argmin(temp)
        
        temp_lccs0 = self.lanecenterlineconnectors[LCC_id]["geometry"][:-1]
        temp_lccs1 = self.lanecenterlineconnectors[LCC_id]["geometry"][1:]
        dist = np.sqrt(np.sum((temp_lccs0 - temp_lccs1) ** 2, 1))
        dist = np.cumsum(dist)
        dist = np.concatenate(([0], dist))
        
        point0 = [x, y]
        
        if lcc_point_id == 0 or (lcc_point_id != self.lanecenterlineconnectors[LCC_id]["geometry"].shape[0] - 1 and 
                                (sum((self.lanecenterlineconnectors[LCC_id]["geometry"][lcc_point_id + 1] - array([x, y])) ** 2, 1) ** 0.5 <
                                sum((self.lanecenterlineconnectors[LCC_id]["geometry"][lcc_point_id - 1] - array([x, y])) ** 2, 1) ** 0.5)):
            point1 = self.lanecenterlineconnectors[LCC_id]["geometry"][lcc_point_id]
            point2 = self.lanecenterlineconnectors[LCC_id]["geometry"][lcc_point_id + 1]
            length = dist[lcc_point_id]
        else:
            point1 = self.lanecenterlineconnectors[LCC_id]["geometry"][lcc_point_id - 1]
            point2 = self.lanecenterlineconnectors[LCC_id]["geometry"][lcc_point_id]
            length = dist[lcc_point_id - 1]
            
        v1 = array(point0) - array(point1)
        v2 = array(point2) - array(point1)
        s = v1[0] * v2[1] - v1[1] * v2[0]
        h = s / (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        length = length + np.dot(v1, v2) / np.dot(v2, v2) * (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        
        res = {}
        res["is_virtual"] = True
        res["LCC_id"] = LCC_id
        res["dist"] = h
        res["len"] = length
        res["RC_id"] = self.lanecenterlineconnectors[LCC_id]["RC_id"]
        res["Section_id"] = self.lanecenterlineconnectors[LCC_id]["Section_id"]
        res["X"] = x
        res["Y"] = y
        return res
    
    def get_point_from_data_lc(self, LC_id, dist, length):
        lc_geometry = self.lanecenterlines[LC_id]["geometry"]
        
        for i in range(lc_geometry.shape[0]):
            if i == lc_geometry.shape[0] - 1 or length < np.linalg.norm(lc_geometry[i + 1] - lc_geometry[i]):
                if i == 0:
                    vector0 = (lc_geometry[i + 1] - lc_geometry[i]) / np.linalg.norm(lc_geometry[i + 1] - lc_geometry[i])
                else:
                    vector0 = (lc_geometry[i] - lc_geometry[i - 1]) / np.linalg.norm(lc_geometry[i] - lc_geometry[i - 1])
                vector1 = np.array([vector0[1], -vector0[0]])
                point = lc_geometry[i] + length * vector0 + dist * vector1
                return point
                    
            length -= np.linalg.norm(lc_geometry[i + 1] - lc_geometry[i])
    
    def get_point_from_data_lcc(self, LCC_id, dist, length):
        lcc_geometry = self.lanecenterlineconnectors[LCC_id]["geometry"]
        
        for i in range(lcc_geometry.shape[0]):
            if i == lcc_geometry.shape[0] - 1 or length < np.linalg.norm(lcc_geometry[i + 1] - lcc_geometry[i]):
                if i == 0:
                    vector0 = (lcc_geometry[i + 1] - lcc_geometry[i]) / np.linalg.norm(lcc_geometry[i + 1] - lcc_geometry[i])
                else:
                    vector0 = (lcc_geometry[i] - lcc_geometry[i - 1]) / np.linalg.norm(lcc_geometry[i] - lcc_geometry[i - 1])
                vector1 = np.array([vector0[1], -vector0[0]])
                point = lcc_geometry[i] + length * vector0 + dist * vector1
                return point
                    
            length -= np.linalg.norm(lcc_geometry[i + 1] - lcc_geometry[i])
    
    def get_data_from_trajectory(self, trajectory):
        res = []
        sec_ids = []
        for point in trajectory:
            res.append(self.get_data_from_location(point[0], point[1]))
            sec_ids.append(None)
        
        last_sec_id = None
        last_index = None
        for i in range(0, len(res)):
            if res[i]["is_virtual"]:
                sec_ids[i] = res[i]["Section_id"]
                if sec_ids[i] == last_sec_id:
                    for j in range(last_index, i):
                        sec_ids[j] = sec_ids[i]
                last_sec_id = sec_ids[i]
                last_index = i
                
        indexes = []
        indexes.append(0)
        if sec_ids[0] is not None:
            indexes.append(0)
        for i in range(1, len(sec_ids)):
            if sec_ids[i] != sec_ids[i - 1]:
                indexes.append(i)
                if sec_ids[i] != None and sec_ids[i - 1] != None:
                    indexes.append(i)
        indexes.append(len(sec_ids))
        if sec_ids[-1] is not None:
            indexes.append(len(sec_ids))
            
            
        # if len(indexes) == 2:
        #     return res
        
        lc_list = []
        for i in range(0, len(indexes) - 1, 2):
            if indexes[i] == indexes[i + 1]:
                lc_list.append(None)
                continue
            
            
            if indexes[i] == 0 and indexes[i + 1] == len(sec_ids):
                R_ids = [self.lanecenterlines[res[0]["LC_id"]]["R_id"]]
            elif indexes[i] == 0:
                R_ids = self.intersections[sec_ids[indexes[i + 1]]]["in_R_id"]
            elif indexes[i + 1] == len(sec_ids):
                R_ids = self.intersections[sec_ids[indexes[i] - 1]]["out_R_id"]
            else:
                sec_id0 = sec_ids[indexes[i] - 1]
                sec_id1 = sec_ids[indexes[i + 1]]
                R_ids = self._get_R_ids_from_sec_ids(sec_id0, sec_id1)
            
            
            min_tot_dist = None
            min_LC_id = None
            
            for R_id in R_ids:
                for LC_id in self.R2LC[R_id]:
                    temp_tot_dist = 0
                    for j in range(indexes[i], indexes[i + 1]):
                        temp = self._get_data_from_location_and_LC_id(trajectory[j][0], trajectory[j][1], LC_id)
                        temp_tot_dist += abs(temp["dist"])
                    if min_tot_dist is None or temp_tot_dist < min_tot_dist:
                        min_tot_dist = temp_tot_dist
                        min_LC_id = LC_id
            
            lc_list.append(min_LC_id)
            
            for j in range(indexes[i], indexes[i + 1]):
                res[j] = self._get_data_from_location_and_LC_id(trajectory[j][0], trajectory[j][1], min_LC_id)
        
        for i in range(1, len(indexes) - 2, 2):
            RC_ids = self._get_RC_ids_from_sec_id(sec_ids[indexes[i]], lc_list[i // 2], lc_list[i // 2 + 1])
        
            if RC_ids == []:
                return None
            
            min_tot_dist = None
            min_LCC_id = None
            
            for RC_id in RC_ids:
                for LCC_id in self.RC2LCC[RC_id]:
                    temp_tot_dist = 0
                    for j in range(indexes[i], indexes[i + 1]):
                        temp = self._get_data_from_location_and_LCC_id(trajectory[j][0], trajectory[j][1], LCC_id)
                        temp_tot_dist += abs(temp["dist"])
                    if min_tot_dist is None or temp_tot_dist < min_tot_dist:
                        min_tot_dist = temp_tot_dist
                        min_LCC_id = LCC_id
                        
            for j in range(indexes[i], indexes[i + 1]):
                res[j] = self._get_data_from_location_and_LCC_id(trajectory[j][0], trajectory[j][1], min_LCC_id)
        
        for i in range(0, len(res)):
            if res[i]['dist'] > 5:
                return None
                
        return res
    
    def _get_RC_ids_from_sec_id(self, sec_id, lc_id0=None, lc_id1=None):
        res = []
        sec = self.intersections[sec_id]
        for rc_id in sec["RC_id"]:
            for lcc_id in self.RC2LCC[rc_id]:
                lcc = self.lanecenterlineconnectors[lcc_id]
                in_lc = self.lanecenterlines[lcc["in_LC_id"]]
                out_lc = self.lanecenterlines[lcc["out_LC_id"]]
                if lc_id0 is not None:
                    lc0 = self.lanecenterlines[lc_id0]
                    if in_lc["R_id"] != lc0["R_id"]:
                        continue
                if lc_id1 is not None:
                    lc1 = self.lanecenterlines[lc_id1]
                    if out_lc["R_id"] != lc1["R_id"]:
                        continue
                res.append(rc_id)
                break
        
        return res
        
    
    def _get_R_ids_from_sec_ids(self, sec_id0, sec_id1):
        res = []
        sec0 = self.intersections[sec_id0]
        sec1 = self.intersections[sec_id1]
        for r_id in sec0["out_R_id"]:
            if r_id in sec1["in_R_id"]:
                res.append(r_id)
        return res
        
    def _get_intersections(self):
        raw_intersections = self.map_api._get_proximity_map_object(self.boundary, SemanticMapLayer.INTERSECTION)
        for raw_intersection in raw_intersections:
            intersection = {}
            intersection["geometry"] = np.asarray(raw_intersection.polygon.exterior.coords.xy).T
            
            # 为了方便，在处理车道的时候对它们进行初始化
            intersection["in_R_id"] = []
            intersection["out_R_id"] = []
            intersection["RC_id"] = []
            
            self.intersections[int(raw_intersection.id)] = intersection
        
    def _get_real_lanes(self):
        raw_lanes = self.map_api._get_proximity_map_object(self.boundary, SemanticMapLayer.LANE)
        for raw_lane in raw_lanes:
            ## centerline
            lane = {}
            lane["geometry"] = np.asarray(raw_lane.baseline_path.linestring.coords.xy).T
            lane["R_id"] = int(raw_lane.get_roadblock_id())
            lane["left_LB_id"] = int(raw_lane.left_boundary.id)
            lane["right_LB_id"] = int(raw_lane.right_boundary.id)
            lane["speed_limit"] = raw_lane.speed_limit_mps
            
            road_block = self.map_api._get_roadblock(int(raw_lane.get_roadblock_id()))
            assert road_block is not None, "车道所在道路不存在，车道id为" + raw_lane.id
            if len(road_block.outgoing_edges) != 0:
                lane["in_Section_id"] = int(road_block.outgoing_edges[0].intersection.id)
                assert int(road_block.outgoing_edges[0].intersection.id) in self.intersections.keys(), \
                        "路口不存在，路口id为" + road_block.outgoing_edges[0].intersection.id
                        
                if int(raw_lane.get_roadblock_id()) not in self.intersections[int(road_block.outgoing_edges[0].intersection.id)]["in_R_id"]:
                    self.intersections[int(road_block.outgoing_edges[0].intersection.id)]["in_R_id"].append(int(raw_lane.get_roadblock_id()))
            else:
                lane["in_Section_id"] = 0
            if len(road_block.incoming_edges) != 0:
                lane["out_Section_id"] = int(road_block.incoming_edges[0].intersection.id)
                assert int(road_block.incoming_edges[0].intersection.id) in self.intersections.keys(), \
                        "路口不存在，路口id为" + road_block.incoming_edges[0].intersection.id
                        
                if int(raw_lane.get_roadblock_id()) not in self.intersections[int(road_block.incoming_edges[0].intersection.id)]["out_R_id"]:
                    self.intersections[int(road_block.incoming_edges[0].intersection.id)]["out_R_id"].append(int(raw_lane.get_roadblock_id()))
            else:
                lane["out_Section_id"] = 0
            
            lane["LCC_id"] = [] # 为了方便，在处理虚拟车道的时候对它进行初始化
            self.lanecenterlines[int(raw_lane.id)] = lane
            
            self.point_geometry = np.append(self.point_geometry, lane["geometry"], axis=0)
            for i in range(0, len(lane["geometry"])):
                temp_point = {}
                temp_point["is_virtual"] = False
                temp_point["L_id"] = int(raw_lane.id)
                temp_point["point_id"] = i
                self.point_data.append(temp_point)
                
                
            if int(raw_lane.get_roadblock_id()) not in self.R2LC.keys():
                self.R2LC[int(raw_lane.get_roadblock_id())] = [int(raw_lane.id)]
            else:
                self.R2LC[int(raw_lane.get_roadblock_id())].append(int(raw_lane.id))
            
            ## left_boundary
            raw_left_boundary = raw_lane.left_boundary
            if int(raw_left_boundary.id) not in self.laneboundaries.keys():
                left_boundary = {}
                left_boundary["geometry"] = np.asarray(raw_left_boundary.linestring.coords.xy).T
                left_boundary["LC_id"] = [int(raw_lane.id)]
                
                self.laneboundaries[int(raw_left_boundary.id)] = left_boundary
            else:
                self.laneboundaries[int(raw_left_boundary.id)]["LC_id"].append(int(raw_lane.id))
                
            ## right_boundary
            raw_right_boundary = raw_lane.right_boundary
            if int(raw_right_boundary.id) not in self.laneboundaries.keys():
                right_boundary = {}
                right_boundary["geometry"] = np.asarray(raw_right_boundary.linestring.coords.xy).T
                right_boundary["LC_id"] = [int(raw_lane.id)]
                
                self.laneboundaries[int(raw_right_boundary.id)] = right_boundary
            else:
                self.laneboundaries[int(raw_right_boundary.id)]["LC_id"].append(int(raw_lane.id))
                
    def _get_virtual_lanes(self):
        lcc_ids = {}
        with open("/nas/user/jqz/codes/SQ2023/nuplan_has_tl_process/result.json", "r", encoding="utf-8") as f:
            content = json.load(f)
            lcc_ids = content[self.map_name]
        
        raw_lane_connectors = self.map_api._get_proximity_map_object(self.boundary, SemanticMapLayer.LANE_CONNECTOR)
        for raw_lane_connector in raw_lane_connectors:
            # assert int(raw_lane_connector.id) not in self.lanecenterlines.keys(), "lane_connector_id 和 lane_id 重复"
            lane_connector = {}
            lane_connector["geometry"] = np.asarray(raw_lane_connector.baseline_path.linestring.coords.xy).T
            lane_connector["RC_id"] = int(raw_lane_connector.get_roadblock_id())
            lane_connector["left_LBC_id"] = int(raw_lane_connector.left_boundary.id)
            lane_connector["right_LBC_id"] = int(raw_lane_connector.right_boundary.id)
            lane_connector["speed_limit"] = raw_lane_connector.speed_limit_mps
            lane_connector["has_traffic_light"] = int(raw_lane_connector.id) in lcc_ids.keys()
            
            road_block_connector = self.map_api._get_roadblock_connector(int(raw_lane_connector.get_roadblock_id()))
            assert road_block_connector.intersection is not None, "虚拟车道所在路口不存在，虚拟车道id为" + raw_lane_connector.id
            lane_connector["Section_id"] = int(road_block_connector.intersection.id)
            if int(raw_lane_connector.get_roadblock_id()) not in self.intersections[int(road_block_connector.intersection.id)]["RC_id"]:
                self.intersections[int(road_block_connector.intersection.id)]["RC_id"].append(int(raw_lane_connector.get_roadblock_id()))
            
            assert len(raw_lane_connector.incoming_edges) == 1, "进入车道数量不为1，虚拟车道id为" + raw_lane_connector.id
            assert len(raw_lane_connector.outgoing_edges) == 1, "退出车道数量不为1，虚拟车道id为" + raw_lane_connector.id
            assert int(raw_lane_connector.incoming_edges[0].id) in self.lanecenterlines.keys(), "对应的进入车道不存在，虚拟车道id为" + raw_lane_connector.id
            assert int(raw_lane_connector.outgoing_edges[0].id) in self.lanecenterlines.keys(), "对应的退出车道不存在，虚拟车道id为" + raw_lane_connector.id
            lane_connector["in_LC_id"] = int(raw_lane_connector.incoming_edges[0].id)
            lane_connector["out_LC_id"] = int(raw_lane_connector.outgoing_edges[0].id)
            self.lanecenterlines[int(raw_lane_connector.incoming_edges[0].id)]["LCC_id"].append(int(raw_lane_connector.id))
            
            self.lanecenterlineconnectors[int(raw_lane_connector.id)] = lane_connector
            
            self.point_geometry = np.append(self.point_geometry, lane_connector["geometry"], axis=0)
            for i in range(0, len(lane_connector["geometry"])):
                temp_point = {}
                temp_point["is_virtual"] = True
                temp_point["L_id"] = int(raw_lane_connector.id)
                temp_point["point_id"] = i
                self.point_data.append(temp_point)
            
            self.LCCstatus[int(raw_lane_connector.id)] = "RED" if lane_connector["has_traffic_light"] else "GREEN"
            
            if int(raw_lane_connector.get_roadblock_id()) not in self.RC2LCC.keys():
                self.RC2LCC[int(raw_lane_connector.get_roadblock_id())] = [int(raw_lane_connector.id)]
            else:
                self.RC2LCC[int(raw_lane_connector.get_roadblock_id())].append(int(raw_lane_connector.id))
            
            ## left_boundary_connector
            raw_left_boundary_connector = raw_lane_connector.left_boundary
            if int(raw_left_boundary_connector.id) not in self.laneboundaryconnectors.keys():
                left_boundary_connector = {}
                left_boundary_connector["geometry"] = np.asarray(raw_left_boundary_connector.linestring.coords.xy).T
                left_boundary_connector["LCC_id"] = [int(raw_lane_connector.id)]
                
                self.laneboundaryconnectors[int(raw_left_boundary_connector.id)] = left_boundary_connector
            else:
                self.laneboundaryconnectors[int(raw_left_boundary_connector.id)]["LCC_id"].append(int(raw_lane_connector.id))
                
            ## right_boundary_connector
            raw_right_boundary_connector = raw_lane_connector.right_boundary
            if int(raw_right_boundary_connector.id) not in self.laneboundaryconnectors.keys():
                right_boundary_connector = {}
                right_boundary_connector["geometry"] = np.asarray(raw_right_boundary_connector.linestring.coords.xy).T
                right_boundary_connector["LCC_id"] = [int(raw_lane_connector.id)]
                
                self.laneboundaryconnectors[int(raw_right_boundary_connector.id)] = right_boundary_connector
            else:
                self.laneboundaryconnectors[int(raw_right_boundary_connector.id)]["LCC_id"].append(int(raw_lane_connector.id))
    
    def generate_json(self):
        res = {}
        res["type"] = "FeatureCollection"
        res["name"] = "Lane_boundary"
        res["crc"] = {"type": "name",
                      "properties": {"name": "urn:ogc:def:crs:EPSG::" + self.NAME_TO_EPSG[self.map_name]}}
        
        
        features = []
        for lb_id in self.laneboundaries.keys():
            lb = self.laneboundaries[lb_id]
            temp = {}
            temp["type"] = "Feature"
            temp["properties"] = {
                "Uid": lb_id,
                "Type": 1,
                "Is_virtual": 1,
                "Form": 3,
                "Isolation": 99,
                "Color": 1,
            }
            temp["geometry"] = {
                "type": "LineString",
                "coordinates": np.append(lb["geometry"], np.zeros([lb["geometry"].shape[0], 1]), axis=1).tolist()
            }
            features.append(temp)
        for lbc_id in self.laneboundaryconnectors.keys():
            lbc = self.laneboundaryconnectors[lbc_id]
            temp = {}
            temp["type"] = "Feature"
            temp["properties"] = {
                "Uid": lbc_id,
                "Type": 1,
                "Is_virtual": 3,
                "Form": 0,
                "Isolation": 99,
                "Color": 0,
            }
            temp["geometry"] = {
                "type": "LineString",
                "coordinates": np.append(lbc["geometry"], np.zeros([lbc["geometry"].shape[0], 1]), axis=1).tolist()
            }
            features.append(temp)
            
        res["features"] = features
        
        
        import os
        if not os.path.exists("map_json/" + self.map_name):
            os.mkdir("map_json/" + self.map_name)
        with open("map_json/" + self.map_name + "/Lane_boundary.json", "w") as f:
            json.dump(res, f)
                     
    def get_lanecenterline_from_LC_id(self, id):
        return self.lanecenterlines[id]
    
    def get_lanecenterlineconnector_from_LCC_id(self, id):
        return self.lanecenterlineconnectors[id]
    
    def get_laneboundary_from_LB_id(self, id):
        return self.laneboundaries[id]
    
    def get_laneboundaryconnector_from_LBC_id(self, id):
        return self.laneboundaryconnectors[id]
    
    def get_intersection_from_Section_id(self, id):
        return self.intersections[id]
    
    def get_lanecenterlines_from_R_id(self, id):
        return self.R2LC[id]
    
    def get_lanecenterlineconnectors_from_RC_id(self, id):
        return self.RC2LCC[id]
    
    def get_status_from_LCCid(self, id):
        return self.LCCstatus[id]
    
    def scaled_line(
        self, Type: Literal['scale', 'fixed_dist'] = 'scale', 
        points: np.ndarray = np.array([]), scale: float = 1.0, fixed_dist = 1.0
    ) -> np.ndarray:
        '''
            Type:
                * scale: use 'scale(float)' to scale the points, 1.0 means no change
                * fixed_dis: use 'fixed_dist' to scale the points
            usage:
                points = np.array([[0,0],[1,0],[2,0],[3,0]])
                
                scaled_line(type='scale', points=points, scale=0.5)
                or
                scaled_line(type='fixed_dist', points=points, fixed_dist=0.5)
        '''
        if type(points) == np.ndarray:
            points = points.tolist()
        
        def dis(point1, point2):
            return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5
        def lower_bound(lst:list, item:float):
            l,r = -1,len(lst)-1
            while l!=r:
                mid = (l+r+1)//2
                if lst[mid] <= item:
                    l = mid
                else:
                    r = mid-1
            return l
        if Type =='scale':
            if scale == 1.0 or len(points)*min(1.0,scale) < 2:
                return points
            else: # scale > 1.0 linearly scale
                ret = []
                length = 0.0
                pre_length = []
                for i in range(len(points)-1):
                    pre_length.append(length)
                    length += dis(points[i], points[i+1])
                pre_length.append(length)
                
                new_num = floor(len(points) * scale)
                udis = length/(new_num-1)
                
                for i in range(new_num-1):
                    loc = i*udis
                    low = lower_bound(pre_length,loc)
                    up = low+1
                    if up >= len(points):
                        break
                    else:
                        seg = dis(points[low], points[up])
                        ret.append(((points[up][0]-points[low][0])*(loc-pre_length[low])/seg + points[low][0], (points[up][1]-points[low][1])*(loc-pre_length[low])/seg + points[low][1]))
                
                ret.append((points[-1][0],points[-1][1]))
                return np.array(ret)
        
        elif Type == 'fixed_dist' and fixed_dist > 0.1:
            ret = []
            length = 0.0
            pre_length = []
            for i in range(len(points)-1):
                pre_length.append(length)
                length += dis(points[i], points[i+1])
            pre_length.append(length)
            
            new_num = floor(length/fixed_dist)+1
            udis = fixed_dist
            
            for i in range(new_num):
                loc = i*udis
                low = lower_bound(pre_length,loc)
                up = low+1
                if up >= len(points):
                    break
                else:
                    seg = dis(points[low], points[up])
                    ret.append(((points[up][0]-points[low][0])*(loc-pre_length[low])/seg + points[low][0], (points[up][1]-points[low][1])*(loc-pre_length[low])/seg + points[low][1]))
            
            ret.append((points[-1][0],points[-1][1]))
            return np.array(ret)
        else:
            raise ValueError('type error')
    
    def plot_lanes(self, lanes, raw_trajectory=None, type=""):
        import matplotlib.pyplot as plt

        for lane in lanes:
            if lane is None:
                continue
            plt.scatter(lane["geometry"][:, 0], lane["geometry"][:, 1], color="black", s=2, linewidths = 1)
        if raw_trajectory is not None:
            raw_trajectory = np.array(raw_trajectory)
            plt.scatter(raw_trajectory[:, 0], raw_trajectory[:, 1], color="red", s=2, linewidths = 1)
        
        plt.savefig("test" + type + ".png", dpi=100)
        plt.close()