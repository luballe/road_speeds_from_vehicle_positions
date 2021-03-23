from numba import jit, njit, types, vectorize, prange
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta 
from sqlalchemy import create_engine
from functools import partial
import io
import gc
import os
import sys
import time
import pygeos
import multiprocessing
import geopandas as gpd
import pandas as pd
import numpy as np
import math
import pytz
import pyodbc
import psutil
import yaml

def execute_query(query, db_engine):
    copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
       query=query, head="HEADER"
    )
    #print(copy_sql)
    #sys.exit()
    conn = db_engine.raw_connection()
    cur = conn.cursor()
    store = io.StringIO()
    cur.copy_expert(copy_sql, store)
    store.seek(0)
    df = pd.read_csv(store)
    return df

def get_veh_positions(sql_tmpl,last_epoch_time):
    # Get credentials
    with open(r'credentials_postgresql.yaml') as file:
      # The FullLoader parameter handles the conversion from YAML
      # scalar values to Python the dictionary format
      credentials = yaml.load(file, Loader=yaml.FullLoader)
      host=credentials['host']
      port=str(credentials['port'])
      db=credentials['db']
      user=credentials['user']
      password=credentials['password']

    
      # Create engine to query Postgresql DB
      #engine = create_engine('postgresql://secretaria_movilidad:rJC4umJa6rNAERmr@localhost:5431/dwhtransmilenio')
      engine = create_engine('postgresql://'+user+':'+password+'@'+host+':'+port+'/'+db)
      # Get SQL query template
      #sql_tmpl = 'get_data_from_db_3.sql'
      query = open(sql_tmpl,'r').read()
      # put parameters in template
      #print('ini_time (-30 secs) = %s' % last_epoch_time)
      #print('query:')
      #print(query)
      #sys.exit()
      query = query.format(time_1=last_epoch_time)
      #print(query)
      #print('Executing query...')
      # Take time 
      #start_time = time.time()
      # Execute query
      return execute_query(query,engine)
      #execute_query(query,engine)
      #return None



@njit(nogil=True,fastmath=True)
def calc_speed_bearing(veh_data_arr,job_num,sub_index_arr):
    #print(job_num)
    #print(sub_index_arr)
    start_index=sub_index_arr[job_num][0]
    end_index=sub_index_arr[job_num][1]
    #print(start_index,end_index)
    num_elems = end_index - start_index
    #print(num_elems)
    indexes  = np.zeros(num_elems,dtype=np.int32)
    speeds   = np.zeros(num_elems,dtype=np.float32)
    bearings = np.zeros(num_elems,dtype=np.int32)

    i=0
    global_index=start_index
    #for i, global_index in zip(range(0, num_elems), range(start_index, end_index)):
    while global_index<end_index:
        indexes[i] = global_index

        #index      = veh_data_arr[global_index][0]
        veh_id_1   = veh_data_arr[global_index][0]
        ruta_id_1  = veh_data_arr[global_index][1]
        viaje_id_1 = veh_data_arr[global_index][2]
        time_1     = veh_data_arr[global_index][3]
        x_1        = veh_data_arr[global_index][4]
        y_1        = veh_data_arr[global_index][5]
        veh_id_2   = veh_data_arr[global_index][6]
        ruta_id_2  = veh_data_arr[global_index][7]
        viaje_id_2 = veh_data_arr[global_index][8]
        time_2     = veh_data_arr[global_index][9]
        x_2        = veh_data_arr[global_index][10]
        y_2        = veh_data_arr[global_index][11]

        if veh_id_1 == veh_id_2 and ruta_id_1 == ruta_id_2 and viaje_id_1 == viaje_id_2 :
            distance = math.sqrt((x_2-x_1)**2 + (y_2-y_1)**2)
            delta_time = time_2 - time_1
            # speed in m/s
            if delta_time > 0:
                speeds[i] = round(distance/delta_time,2)
            else:
                speeds[i] = -1
            # Bearing
            if (x_1==x_2 and y_1==y_2):
                if(i==0):
                    bearings[i] =  0
                else:
                    bearings[i] =  bearings[i-1]
            else:
                bearings[i] = round((90 - math.degrees(math.atan2(y_2 - y_1, x_2 - x_1))) % 360,2)
        else:
            speeds[i] = -1
            bearings[i] = -1
        i = i+1
        global_index = global_index+1
    #print(indexes)
        
    return indexes, speeds, bearings

@njit(nogil=True,fastmath=True)
def calc_speeds_bearings_wrapper():
    num_processors = 5
    num_positions=len(veh_data_arr)
    if num_positions > num_processors:
        block_size=int(math.floor(num_positions/num_processors))
        #print(block_size)

        sub_index_arr= np.zeros((num_processors,2),dtype=np.int32)
        result_arr= np.zeros((num_positions,2),dtype=np.float32)

        start_index = 0
        end_index = block_size

        for i in range(num_processors):
            sub_index_arr[i][0] = int(start_index)
            sub_index_arr[i][1] = int(end_index)
            start_index = start_index + block_size 
            end_index = end_index + block_size
        #a = num_positions%num_processors
        #if ((a) > 0):
        sub_index_arr[num_processors-1][1] = num_positions
    else:
        start_index = 0
        end_index = num_positions

        sub_index_arr= np.zeros((1,2),dtype=np.int32)
        result_arr= np.zeros((num_positions,2),dtype=np.float32)
        sub_index_arr[0][0] = 0
        sub_index_arr[0][1] = num_positions
        num_processors = 1
    
    #print(num_processors)
    #print(sub_index_arr)
    #print(block_size)
    
    for job_num in range(num_processors):
        index_arr_res,speeds_arr_res,bearings_arr_res = calc_speed_bearing(veh_data_arr,job_num,sub_index_arr)
        #print(len(index_arr_res))
        for i in range(len(index_arr_res)):
            #result_arr[index_arr_res[i]][0]=index_arr_res[i]
            result_arr[index_arr_res[i]][0]=speeds_arr_res[i]
            result_arr[index_arr_res[i]][1]=bearings_arr_res[i]

    return result_arr

def get_geometric_lines(data):
    #print(data)
    x_1      = data[0]
    y_1      = data[1]
    x_2      = data[2]
    y_2      = data[3]
    return LineString([Point(x_1, y_1), Point(x_2, y_2)])

def intersect_lines_with_polygons(data,ini,end):
    #print(ini,end)
#    _av_polygons=data[0]
    return gpd.overlay(veh_data[ini:end], data[0], how='intersection').explode().reset_index(drop=True)


@njit(nogil=True,fastmath=True)
def fast_calc_vec_length_time(mix_arr,vectors,num_elems):

    vec_lengths   = np.zeros(num_elems,dtype=np.float32)
    vec_times     = np.zeros(num_elems,dtype=np.float32)
    
    i = 0
    while i < num_elems:
        seg_bearing  = mix_arr[i][0]
        veh_bearing  = mix_arr[i][1]
        veh_speed_ms = mix_arr[i][2]
        x1           = vectors[i][0]
        y1           = vectors[i][1]
        x2           = vectors[i][2]
        y2           = vectors[i][3]
        #print(seg_bearing,veh_bearing,veh_speed_ms,x1,y1,x2,y2)        

        # Normalize angles to be between 0 and 359
        seg_bearing = seg_bearing%360
        veh_bearing = veh_bearing%360
        
        # Get the absolute offset angles
        start_angle = seg_bearing - offset_degrees
        end_angle = seg_bearing + offset_degrees

        bearing_offset_ok = False
        if start_angle >= 0:
            if end_angle < 360:
                if start_angle <= veh_bearing <= end_angle:
                    bearing_offset_ok = True
            else:
                new_end_angle = end_angle-360
                if (start_angle <= veh_bearing < 360) or 0 <= veh_bearing <= new_end_angle:
                    bearing_offset_ok = True
        else:
            new_start_angle = start_angle+360
            if (new_start_angle <= veh_bearing < 360) or (0 <= veh_bearing <= end_angle):
                bearing_offset_ok = True

        if bearing_offset_ok:
            if x1 == 0:
                vec_len = 0
            else:
                vec_len = round(math.sqrt((x2-x1)**2 + (y2-y1)**2),2)
            if veh_speed_ms > 0:
                vec_time = round(vec_len / veh_speed_ms,2)
            else:
                vec_time = 0
            vec_lengths[i] = vec_len
            vec_times[i] = vec_time
            
        else:
            vec_lengths[i] = -1
            vec_times[i] = -1

        i = i + 1
    return vec_lengths,vec_times

def calc_vec_len_time(job_num, index_arr, df, vec_lengths, vec_times):
    start_index = index_arr[job_num][0]
    end_index = index_arr[job_num][1]
    num_elems = end_index - start_index

    mix_arr = df[start_index:end_index][['BEARING', 'veh_bearing','veh_speed_ms']].to_numpy(dtype=np.float32)
    vectors= np.zeros((num_elems,4),dtype=np.float32)
    
    vecs = df[start_index:end_index].geometry.values
    i = 0
    for line in vecs:
        if( len(line.coords) == 2):
            vectors[i][0] = line.coords[0][0]
            vectors[i][1] = line.coords[0][1]
            vectors[i][2] = line.coords[1][0]
            vectors[i][3] = line.coords[1][1]
        else:
            vectors[i][0] = 0
            vectors[i][1] = 0
            vectors[i][2] = 0
            vectors[i][3] = 0
        i = i + 1

    a,b = fast_calc_vec_length_time(mix_arr,vectors,num_elems)
    vec_lengths[start_index:end_index] = a
    vec_times[start_index:end_index] = b
    #print(len(a),len(b))


def calc_vector_length_time_wrapper(num_proc, num_pos, df, vec_lengths, vec_times):
    #print(num_processors)

    num_processors = num_proc
    # TEST
    #num_processors = 1
    
    num_positions = num_pos

    if num_positions > num_processors:
        block_size=int(math.floor(num_positions/num_processors))
        #print(block_size)

        sub_index_arr= np.zeros((num_processors,2),dtype=np.int32)

        start_index = 0
        end_index = block_size

        for i in range(num_processors):
            sub_index_arr[i][0] = int(start_index)
            sub_index_arr[i][1] = int(end_index)
            start_index = start_index + block_size 
            end_index = end_index + block_size
        #if (num_positions % num_processors) > 0 :
        sub_index_arr[num_processors-1][1] = num_positions
    else:
        start_index = 0
        end_index = num_positions
        sub_index_arr= np.zeros((1,2),dtype=np.int32)
        sub_index_arr[0][0] = 0
        sub_index_arr[0][1] = num_positions
        num_processors = 1
    
    #print(num_processors)
    #print(sub_index_arr)
    #print(block_size)
    processes=[]

    for job_num in range(num_processors):
        p = multiprocessing.Process(target=calc_vec_len_time,args=[job_num,sub_index_arr,df,vec_lengths,vec_times])
        p.start()
        processes.append(p)
    for process in processes:
        process.join()


def calc_quarter(data):
    #print(data)
    timestamp = data[0]
    result = []

    dt = datetime.fromtimestamp(timestamp)
    date_and_time = dt + timedelta(hours = 5)
    
    result   = [
        get_quarter(date_and_time),
        date_and_time.strftime('%Y-%m-%d')
    ]
    return result

#@jit(nopython=True)
def get_quarter(date_time):
    result = ''
    if date_time.minute < 15:
        result = f'{date_time.hour:02}'+':00-'+f'{date_time.hour:02}'+':15' 
    elif date_time.minute >= 15 and date_time.minute < 30:
        result =f'{date_time.hour:02}'+':15-'+f'{date_time.hour:02}'+':30'
    elif date_time.minute >= 30 and date_time.minute < 45:
        result =f'{date_time.hour:02}'+':30-'+f'{date_time.hour:02}'+':45'
    elif date_time.minute >= 45:
        if date_time.hour < 23:
            result =f'{date_time.hour:02}'+':45-'+f'{(date_time.hour+1):02}'+':00'
        else:
            result =f'{date_time.hour:02}'+':45-00:00'
    return result
    

total_start_time = time.time()
if __name__ == '__main__':
    #start_time_total = time.time()
    # Set optimizations
    # Number of processors
    num_processors=multiprocessing.cpu_count()
    # Use pygeos as posssible to speed things up
    gpd.options.use_pygeos = True
    # max offset degrees that a vehicle brearing match a segment bearing
    offset_degrees = 15
    # Get info from BD (or CSV)
    get_veh_data_from_bd = 1
    filename=''
    # Get vehicle info
    if get_veh_data_from_bd == 1:
        # Execute query
        last_time_file='last_time.txt'
        start_time = time.time()
        with open(last_time_file) as f:
            ini_time_str = f.readline()
        ini_time=int(ini_time_str)
        print('Ini Time: %s ' % ini_time)
        #sys.exit()
        print('Running query...')
        veh_data=get_veh_positions('get_data_from_db.sql',(ini_time-30))
        print('OK query! --- %s seconds ---' % (time.time() - start_time)) 
        #sys.exit()
        print('Num of records retrieved: %s' % len(veh_data))
        if len(veh_data) > 0:
            column = veh_data["datetime1"]
            end_time = column.max()
            print('End Time: %s ' % end_time)
            if ini_time == end_time:
                print('No records... Exit!')
                sys.exit()
            f = open(last_time_file, "w")
            f.write(str(end_time))
            f.close()
            path='/monitoreo/sitp_speeds/positions/'
            filename=str(ini_time)+"_"+str(end_time)+".csv"
            print('Saving data...')
            start_time = time.time()
            veh_data.to_csv(path+filename, index=False)
            print('Data Saved! --- %s seconds ---' % (time.time() - start_time))
        else:
            print('No data...')
            sys.ext()
    else:
        filename=sys.argv[1]
        path='/monitoreo/sitp_speeds/positions/'
        read_filename=path+filename
        start_time = time.time()
        print('Reading file %s...' % read_filename)
         # Read csv file
        #bus_positions = pd.read_csv("bus_positions.csv",header = None)
        veh_data = pd.read_csv(read_filename)
        print(len(veh_data))
        print('CSV read! --- %s seconds ---' % (time.time() - start_time))



    start_time = time.time()
    print('Copying and shifting dataframe...')
    # Create a copy of the bus positions dataframe, shift it one position down and paste it at the right of the original
    veh_data_copy = veh_data.copy()
    veh_data = veh_data.shift(1,fill_value=0)
    # Add and additional index to paste it into the numpy array
    #index = pd.Series([i for i in range(len(veh_data))])
    # Concat all dataframes in one
    veh_data = pd.concat([veh_data,veh_data_copy], axis=1)
    del veh_data_copy
    gc.collect()
    #del index
    veh_data = veh_data.reset_index(drop=True)
    veh_data.columns = ['veh_id_1','ruta_id_1','viaje_id_1','timestamp_1','x_1','y_1','veh_id_2','ruta_id_2','viaje_id_2','timestamp_2','x_2','y_2']
    print('Copied and shift! --- %s seconds ---' % (time.time() - start_time)) 
    
    # Define numpy array for numba processing
    veh_data_arr = veh_data.to_numpy(dtype=np.int32)
    #veh_data_arr = veh_data[0:20].to_numpy(dtype='int32')


    # compile calc_speed_bearing
    # array of two positions
    a = np.zeros((0, 2),dtype=np.int32)
    _ = calc_speed_bearing(veh_data_arr,0,a)

    start_time = time.time()
    print('Calculating speeds and bearings...')
    # calculate speeds and bearings
    speeds_bearings=pd.DataFrame(calc_speeds_bearings_wrapper(),columns=['veh_speed_ms','veh_bearing'])
    del(veh_data_arr)
    gc.collect()
    print('Speeds and bearings calculated! --- %s seconds ---' % (time.time() - start_time))

    # Concat speeds and bearing to vehicle data
    start_time = time.time()
    print('Concat speeds and bearing to vehicle data...')
    veh_data = pd.concat([veh_data,speeds_bearings], axis=1)
    veh_data = veh_data[(veh_data["veh_bearing"] != -1) & (veh_data["veh_speed_ms"] != -1) & (veh_data["veh_speed_ms"] < 17)]
    veh_data.drop(['timestamp_1','veh_id_2', 'ruta_id_2','viaje_id_2'], axis=1, inplace=True)
    # Rename dataframe
    #bus_speeds = bus_positions
    del(speeds_bearings)
    gc.collect()
    #print(veh_data)
    print('Speeds and bearings concatenated! --- %s seconds ---' % (time.time() - start_time))


    # JUST FOR TEST!!!!!!
    #veh_data = veh_data[0:1000]


    # Assign geometry to df with parallelism
    # Get lines from points
    start_time = time.time()
    print('Assigning Linestring to veh_data...')
    __slots__ = ['num_processors']
    with multiprocessing.Pool(num_processors) as p:
        lines = p.map(get_geometric_lines, zip(
            veh_data['x_1'],
            veh_data['y_1'],
            veh_data['x_2'],
            veh_data['y_2']
        ))
    veh_data = gpd.GeoDataFrame(veh_data,geometry=lines)
    veh_data = veh_data.set_crs(epsg=32618)
    del(lines)
    gc.collect()
    #print(veh_data)
    print('LineString assigned!')
    print("--- %s seconds ---" % (time.time() - start_time))

    # Load shapefile
    corr_polygons = gpd.read_file("shape_natalie_fixed_UTM_18N/shape_natalie_fixed_UTM_18N.shp")
    #print(corr_polygons)

    # Create an array of geodataframes (of num_processors positions)
    df_polys = []
    block_size = int(len(corr_polygons) / num_processors)
    
    ini_index=0
    end_index=block_size
    for i in range(num_processors):
        if i == (num_processors-1):
            if len(corr_polygons) % num_processors > 0:
                end_index = len(corr_polygons)
        rows = corr_polygons.iloc[ ini_index:end_index , : ]
        d = {
            'POLY_INDEX': [i for i in range(ini_index,end_index)],
            'ID_CORR': rows['ID_CORR'],
            'CORREDOR': rows['CORREDOR'],
            'ID_TRAMO': rows['ID_TRAMO'],
            'TRAMO_INI': rows['TRAMO_INI'],
            'TRAMO_FIN': rows['TRAMO_FIN'],
            'ID_SEGMENT': rows['ID_SEGMENT'],
            'SEG_LEN': rows['SEG_LEN'],
            'BEARING': rows['BEARING'],
            'SENTIDO': rows['SENTIDO'] 
        }
        new_polys = gpd.GeoDataFrame(d,geometry=rows['geometry'])
        df_polys.append(new_polys) 
        
        ini_index=ini_index+block_size
        end_index=end_index+block_size

    # Intersect lines with polygons 
    start_time = time.time()
    print('Intersecting lines with polygons ...')

    #num_processors = 1
   
    num_elem = len(veh_data)

    print('num_elems = %s ' % num_elem)
    
    # Max number of vectors to be intersected with polygons
    # It depends on the system RAM available
    #max_items_per_iteration = 5000000
    max_items_per_iteration = math.ceil(psutil.virtual_memory().available / 1200000000) * 100000
    print('Max items per iteration: ' + str(max_items_per_iteration))

    num_iters=int(math.floor(num_elem/max_items_per_iteration))
    if (num_elem >0 and num_iters == 0):
        num_iters = 1
    elif (num_elem >0 and num_iters > 0):
        if(num_elem % max_items_per_iteration > 0):
            num_iters = num_iters + 1
    print('num_iters = %s ' % num_iters)
    
    start_index = 0
    end_index = max_items_per_iteration
    sub_index_arr= np.zeros((num_iters,2),dtype=np.int32)
    
    for i in range(num_iters):
        sub_index_arr[i][0] = int(start_index)
        sub_index_arr[i][1] = int(end_index)
        start_index = start_index + max_items_per_iteration
        end_index = end_index + max_items_per_iteration
    sub_index_arr[num_iters-1][1] = num_elem
    print('sub_index_arr:')
    print(sub_index_arr)
    
    gc.collect()
    result_list = []
    for i in range(num_iters):
        iter_start_time = time.time()
        print('---- iter No : %s ' % (i+1))
        with multiprocessing.Pool(num_processors) as p:
            ini = sub_index_arr[i][0]
            end = sub_index_arr[i][1]
            prod_x=partial(intersect_lines_with_polygons, ini=ini,end=end)
            temp_list = p.map(prod_x,zip(df_polys))
            result_list.append(temp_list)
        print('Iter elapsed time:  --- %s seconds ---' % (time.time() - iter_start_time))

    print('Lines and polygons intersected! --- %s seconds ---' % (time.time() - start_time))
    print('result_list length:')
    print(len(result_list))

    gc.collect()
    start_time = time.time()
    print('Joining new vectors into one dataframe...')
    del(df_polys)
    # Join polys into the veh_data dataframe
    veh_data = gpd.GeoDataFrame()
    total_vec = 0
    for i in range(num_iters):
        print('iter= %s' % (i+1))
        temp = result_list[i]
        for df in temp:
           num_vec = len(df)
           print(num_vec)
           total_vec = total_vec + num_vec
           veh_data = veh_data.append(df,ignore_index=True)
    print('Total Vectors : %s ' % total_vec)
    print('New vectors joined ! --- %s seconds ---' % (time.time() - start_time))

    # Compile fast_calc_vec_length_time
    param1 = np.zeros((1,3),dtype=np.float32)
    param2 = np.zeros((1,4),dtype=np.float32)
    param3 = 0 
    a,b = fast_calc_vec_length_time(param1,param2,param3)
    del(a)
    del(b)

    gc.collect()
    start_time = time.time()
    print('Getting vectors lengths and times ...')
    # Run calculation of vector's length and time
    veh_data_2 = veh_data
    #veh_data_2 = veh_data[0:20]
    
    #Setup output arrays
    size = len(veh_data_2)
    vec_lengths = multiprocessing.RawArray( 'f', size )
    vec_times = multiprocessing.RawArray( 'f', size )
    
    #Do the actual processing for calcutating length and times of each vector
    calc_vector_length_time_wrapper(num_processors, size, veh_data_2, vec_lengths, vec_times)
    
    # Convert the answer to numpy array
    temp_lengths = np.ctypeslib.as_array(vec_lengths)
    temp_times = np.ctypeslib.as_array(vec_times)
    
    # Convert the numpy array to dataframe
    seg_lengths=pd.DataFrame(data=temp_lengths.reshape(len(temp_lengths),1),columns=['vec_length'])
    seg_times=pd.DataFrame(data=temp_times.reshape(len(temp_times),1),columns=['vec_time'])
    
    # Delete temporal arrays
    del(veh_data_2)
    del(vec_lengths)
    del(vec_times)
    del(temp_lengths)
    del(temp_times)
    #seg_lengths=pd.DataFrame(data=vec_lengths,columns=['seg_length'])
    #seg_times=pd.DataFrame(vec_times,columns=['seg_length'])
    
    # Append length and time dataframe to the vehicle data
    veh_data = pd.concat([veh_data,seg_lengths,seg_times], axis=1)
    del(seg_lengths)
    del(seg_times)
    print('Vectors lengths and times appended to veh_data! --- %s seconds ---' % (time.time() - start_time))

    gc.collect()
    # Remove empty lengths (and times)
    veh_data = veh_data[(veh_data["vec_length"] != -1)]
    veh_data.drop(['x_1','y_1', 'x_2','y_2','veh_bearing','POLY_INDEX','BEARING','SEG_LEN','geometry'], axis=1, inplace=True)
    veh_data = veh_data.reset_index(drop=True)

    start_time = time.time()
    print('Getting vectors dates and quarters ...')
    # Get the date and quarter from timestamp
    with multiprocessing.Pool(num_processors) as p:
        quarters_array = p.map(calc_quarter, zip(
                veh_data['timestamp_2']
            ))

    # Create a dataframe from date and quarters 
    df_quarters = gpd.GeoDataFrame(quarters_array, columns = ['quarter','date'])
    del(quarters_array)

    # Append date and quarters to vehicle's data
    veh_data = pd.concat([veh_data,df_quarters], axis=1)
    del(df_quarters)
    #print(veh_data)
    print('Vectors dates and quarters appended to veh_data! --- %s seconds ---' % (time.time() - start_time))    

    start_time = time.time()
    print('Groupping and aggregations ...')
    # Do the grouping and aggregations
    veh_data = veh_data.groupby(
       ['date', 'quarter','ruta_id_1','CORREDOR','TRAMO_INI', 'TRAMO_FIN','SENTIDO']
    ).agg(
        {
             'vec_length':sum,    # Sum duration per group
             'vec_time': sum  # get the count of networks
        }
    )

    # Calculate the speeds
    veh_data['speed_ms'] = veh_data.vec_length/veh_data.vec_time
    #print(veh_data)
    # Remove length and time columns
    veh_data.drop(['vec_length','vec_time'], axis=1, inplace=True)
    #print(veh_data)
    # Remove infinite and nan records
    veh_data = veh_data.replace([np.inf, -np.inf], np.nan)
    veh_data = veh_data.dropna(how='all')

    print('Done groupping and cleanning veh_data! --- %s seconds ---' % (time.time() - start_time))
    print('Total records: ')
    print(len(veh_data))
    save_path='/monitoreo/sitp_speeds/result_speeds/'
    #path+str(ini_time)+"_"+str(end_time)+".csv"
    temp_filename=save_path+'temp_'+filename
    #print(veh_data)
    veh_data.to_csv(temp_filename, header=False)
    final_filename=save_path+filename
    cmd_str = 'mv '+temp_filename+' '+final_filename
    #print(cmd_str)
    os.system(cmd_str)
   
    print('inserting records in database...') 
    # Get credentials
    with open(r'credentials_sqlserver.yaml') as file:
      # The FullLoader parameter handles the conversion from YAML
      # scalar values to Python the dictionary format
      credentials = yaml.load(file, Loader=yaml.FullLoader)
      host=credentials['host']
      port=str(credentials['port'])
      db=credentials['db']
      user=credentials['user']
      password=credentials['password']

      # Configure driver
      cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+host+';DATABASE='+db+';UID='+user+';PWD='+password)
      # Create a cursor from the connection
      cursor = cnxn.cursor()
  
      num_records = len(veh_data)
  
      # Do the inserts
      i=1
      for index, row in veh_data.iterrows():
          cursor.execute("insert into velocidades_sitp_carril_preferencial (date,quarter,ruta_id,corredor,tramo_ini,tramo_fin,sentido,vel_ms) values ('"+index[0]+"','"+index[1]+"',"+str(index[2])+",'"+index[3]+"','"+index[4]+"','"+index[5]+"','"+index[6]+"',"+str(row['speed_ms'])+")")
          if i%1000 == 0:
              cnxn.commit()
              print(' %s / %s' % (i,num_records)) 
          i=i+1
  
  
      #commit the transaction
      cnxn.commit()

    print('Done!!!')
    print('TOTAL TIME : --- %s seconds ---' % (time.time() - total_start_time))
    
    
