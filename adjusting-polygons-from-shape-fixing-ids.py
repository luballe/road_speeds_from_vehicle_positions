import pandas as pd
import geopandas as gpd
import numpy as np
import concurrent.futures
from datetime import datetime
from shapely.geometry import Point, LineString
from math import degrees, atan2

def get_bearing(x1, y1, x2, y2):
    angle = degrees(atan2(y2 - y1, x2 - x1))
    #bearing1 = (angle + 360) % 360
    bearing2 = (90 - angle) % 360
    return bearing2

if __name__ == '__main__':
    start_time_total = time.time()
    av_polygons = gpd.read_file("zip://shape_natalie_fixed_UTM_18N.zip!shape_natalie_fixed_UTM_18N.shp")
    #av_polygons.to_csv('av_polygons.csv')
    av_polygons.sort_values(by=['OBJECTID'],inplace=True)
    _crs = av_polygons.crs
    av_polygons

    result = gpd.GeoDataFrame()

    prev_corr = ''
    curr_corr = ''
    for index, row in av_polygons.iterrows():
        #print(row)
        #print(index)
        curr_corr = row['CORREDOR']
        if curr_corr != prev_corr:
            id_tramo = 1
        row_data = {
            "CORREDOR": row['CORREDOR'],
            "ID_CORR": row['COD_CORR'],
            "TRAMO_INI": row['TRAMO_INI'],
            "TRAMO_FIN": row['TRAMO_FIN'],
            "ID_TRAMO": id_tramo,
            "SENTIDO": row['SENTIDO'],
            "geometry": row['geometry'],
        }
        
        result = result.append(row_data,ignore_index=True)

        if row['SENTIDO'] == 'NS':
            sentido = 'SN'
        elif row['SENTIDO'] == 'SN':
            sentido = 'NS'
        elif row['SENTIDO'] == 'WE':
            sentido = 'EW'
        elif row['SENTIDO'] == 'EW':
            sentido = 'WE'
            
        tramo_ini = row['TRAMO_FIN']
        tramo_fin = row['TRAMO_INI']

        tuples = row['geometry'].coords
        inv_tuples = []
    #    print(len(tuples))
        i=(len(tuples)-1)
        while i >= 0:
            inv_tuples.append(tuples[i])
            i=i-1 

        #Cra 15 is only one way to north
        if int(row['COD_CORR']) != 4 and int(row['COD_CORR']) != 15: 
            id_tramo=id_tramo+1
            row_data = {
                "CORREDOR": row['CORREDOR'],
                "ID_CORR": row['COD_CORR'],
                "TRAMO_INI": tramo_ini,
                "TRAMO_FIN": tramo_fin,
                "ID_TRAMO": id_tramo,
                "SENTIDO": sentido,
                "geometry": LineString(inv_tuples),
            }

            result = result.append(row_data,ignore_index=True)
        prev_corr = curr_corr
        id_tramo=id_tramo+1

    result['ID_CORR'] = result['ID_CORR'].astype(np.int32)
    result['ID_TRAMO'] = result['ID_TRAMO'].astype(np.int32)
    av_polygons = result.copy()
    del result
    av_polygons

    result = gpd.GeoDataFrame()

    min_dist = 1
    max_dist = 100

    prev_seg = ''
    curr_seg = ''
    for index, row in av_polygons.iterrows():
        curr_seg = row['TRAMO_INI'] + row['TRAMO_FIN']
        if curr_seg != prev_seg:
            id_segment = 1
        tuples = row['geometry'].coords
        #seg_lnstr = []
        for i in range(len(tuples)-1):
            pt_ini = tuples[i]
            pt_end = tuples[i+1]
    #        pt_new = tuples[i]
            seg_lnstr = LineString([pt_ini,pt_end])
            dist = math.sqrt((pt_end[0]-pt_ini[0])**2 + (pt_end[1]-pt_ini[1])**2)
            if dist < max_dist and dist > min_dist:
                row_data = {
                    "CORREDOR": row['CORREDOR'],
                    "ID_CORR": row['ID_CORR'],
                    "TRAMO_INI": row['TRAMO_INI'],
                    "TRAMO_FIN": row['TRAMO_FIN'],
                    "ID_TRAMO": row['ID_TRAMO'],
                    "SENTIDO": row['SENTIDO'],
                    "ID_SEGMENT": id_segment,
                    "SEG_LEN": round(dist,2),
                    "BEARING": get_bearing(pt_ini[0], pt_ini[1], pt_end[0], pt_end[1]),
                    "geometry": seg_lnstr
                }
                result = result.append(row_data,ignore_index=True)
                id_segment = id_segment + 1
            else:
                orig_dist = dist
                num_chunks = int(dist / max_dist)
    #            print(dist,num_chunks)
                while num_chunks > 0:# and new_dist > 10:
                    pt_new = seg_lnstr.interpolate(max_dist).coords[0]
    #                print(type(pt_new))
    #                print(type(pt_ini))
                    seg_dist = math.sqrt((pt_new[0]-pt_ini[0])**2 + (pt_new[1]-pt_ini[1])**2)
                    seg_lnstr = LineString([pt_ini,pt_new])
                    row_data = {
                        "CORREDOR": row['CORREDOR'],
                        "ID_CORR": row['ID_CORR'],
                        "TRAMO_INI": row['TRAMO_INI'],
                        "TRAMO_FIN": row['TRAMO_FIN'],
                        "ID_TRAMO": row['ID_TRAMO'],
                        "SENTIDO": row['SENTIDO'],
                        "ID_SEGMENT": id_segment,
                        "SEG_LEN": round(seg_dist,2),
                        "BEARING": get_bearing(pt_ini[0], pt_ini[1], pt_new[0], pt_new[1]),
                        "geometry": seg_lnstr
                    }
                    result = result.append(row_data,ignore_index=True)
                    seg_lnstr = LineString([pt_new,pt_end])
                    pt_ini = pt_new
                    num_chunks = num_chunks - 1
                    id_segment = id_segment + 1
                    
                if orig_dist % max_dist > 0:# and new_dist > 10:
                    #pt_new = seg_lnstr.interpolate(max_dist).coords[0]
                    seg_dist = math.sqrt((pt_end[0]-pt_new[0])**2 + (pt_end[1]-pt_new[1])**2)
                    if seg_dist > min_dist:
                        seg_lnstr = LineString([pt_new,pt_end])
                        row_data = {
                            "CORREDOR": row['CORREDOR'],
                            "ID_CORR": row['ID_CORR'],
                            "TRAMO_INI": row['TRAMO_INI'],
                            "TRAMO_FIN": row['TRAMO_FIN'],
                            "ID_TRAMO": row['ID_TRAMO'],
                            "SENTIDO": row['SENTIDO'],
                            "ID_SEGMENT": id_segment,
                            "SEG_LEN": round(seg_dist,2),
                            "BEARING": get_bearing(pt_new[0], pt_new[1], pt_end[0], pt_end[1]),
                            "geometry": seg_lnstr
                        }
                        result = result.append(row_data,ignore_index=True)
                        id_segment = id_segment + 1
            #print(dist)
            prev_seg = curr_seg
        
    result['ID_CORR'] = result['ID_CORR'].astype(np.int32)
    result['ID_TRAMO'] = result['ID_TRAMO'].astype(np.int32)
    result['ID_SEGMENT'] = result['ID_SEGMENT'].astype(np.int32)
    result['BEARING'] = result['BEARING'].astype(np.int16)
    #result
    av_polygons = result[['ID_CORR','CORREDOR','ID_TRAMO','TRAMO_INI','TRAMO_FIN','ID_SEGMENT','SEG_LEN','BEARING','SENTIDO','geometry']]
    #av_polygons = result.copy()
    del result
    av_polygons

    rectangles = av_polygons.geometry.apply(lambda g: g.buffer(70, cap_style=2))
    av_polygons['geometry']=rectangles
    #av_polygons.crs=_crs
    av_polygons = av_polygons.set_crs(_crs)
    av_polygons = av_polygons.to_crs(epsg=32618)
    av_polygons.to_file("shape_natalie_fixed_UTM_18N/shape_natalie_fixed_UTM_18N.shp")
