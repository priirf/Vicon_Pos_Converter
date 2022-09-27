import paho.mqtt.client as mqtt #import the client1
import json
import numpy as np
import pandas as pd
from time import sleep, time
import seaborn as sns
import matplotlib.pylab as plt
from modules_lib.model import ModelWrapper
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 


broker_address = "129.217.152.1"
data = []
rssi = []
magnetometer = []
accelerometer = []
gyroscope = []
timestamp = np.zeros((1,1))

rssi_avg = 0
magneto_avg = 0
accelero_avg = 0
gyro_avg = 0

count = 0
cnt = 0
count_plt = 0
cond = True

model_file_i = 'data/models/model_coords_old'
model_wrapper = ModelWrapper(model_file_i)


KEYS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']

rssi_mat = np.zeros((23,15))
data_mag = np.zeros((23,15))
# X_input = np.zeros([1, 23, 15, 10])
# t_input = np.zeros([345, 1])

timestamps_array = []

X_data = np.zeros([23, 15, 10])
t_i_array = np.zeros((23,15))
t_data = np.zeros((345, 1))
t_batch_i_old_arr = np.zeros((23,15))

# rssi_mat = np.zeros((15,23))
# data_mag = np.zeros((15,23))
data_mat = []

RPi_IPs = [
            {"column_num": 1, "ip_addr": "129.217.152.74", "mac_id": "b8:27:eb:41:99:a0", "hostname": "raspberrypi"},
            {"column_num": 2, "ip_addr": "129.217.152.111", "mac_id": "b8:27:eb:c0:fd:6a", "hostname": "raspberrypi"},
            {"column_num": 3, "ip_addr": "129.217.152.79", "mac_id": "b8:27:eb:18:92:c7", "hostname": "raspberrypi"},
            {"column_num": 4, "ip_addr": "129.217.152.54", "mac_id": "b8:27:eb:53:f2:33", "hostname": "raspberrypi"},
            {"column_num": 5, "ip_addr": "129.217.152.86", "mac_id": "b8:27:eb:e7:6f:dc", "hostname": "raspberrypi"},
            # {"column_num": 6, "ip_addr": "129.217.152.110", "mac_id": "b8:27:eb:9b:69:9a", "hostname": "raspberrypi"},
            {"column_num": 6, "ip_addr": "129.217.152.89", "mac_id": "b8:27:eb:38:4b:07", "hostname": "raspberrypi"},
            {"column_num": 7, "ip_addr": "129.217.152.84", "mac_id": "b8:27:eb:1b:cf:26", "hostname": "raspberrypi"},
            {"column_num": 8, "ip_addr": "129.217.152.119", "mac_id": "b8:27:eb:6d:0e:53", "hostname": "raspberrypi"},
            {"column_num": 9, "ip_addr": "129.217.152.77", "mac_id": "b8:27:eb:b7:a3:b7", "hostname": "raspberrypi"},
            {"column_num": 10, "ip_addr": "129.217.152.118", "mac_id": "b8:27:eb:be:dc:32", "hostname": "raspberrypi"},
            {"column_num": 11, "ip_addr": "129.217.152.69", "mac_id": "b8:27:eb:ff:a4:48", "hostname": "raspberrypi"},
            {"column_num": 12, "ip_addr": "129.217.152.59", "mac_id": "b8:27:eb:a9:7d:4d", "hostname": "raspberrypi"},
            {"column_num": 13, "ip_addr": "129.217.152.85", "mac_id": "b8:27:eb:c4:f8:c7", "hostname": "raspberrypi"},
            {"column_num": 14, "ip_addr": "129.217.152.48", "mac_id": "b8:27:eb:e4:43:6d", "hostname": "raspberrypi"},
            {"column_num": 15, "ip_addr": "129.217.152.63", "mac_id": "b8:27:eb:98:69:6e", "hostname": "raspberrypi"},
            {"column_num": 16, "ip_addr": "129.217.152.50", "mac_id": "b8:27:eb:75:c7:a2", "hostname": "raspberrypi"},
            {"column_num": 17, "ip_addr": "129.217.152.37", "mac_id": "b8:27:eb:09:3d:77", "hostname": "raspberrypi"},
            {"column_num": 18, "ip_addr": "129.217.152.60", "mac_id": "b8:27:eb:05:d8:4d", "hostname": "raspberrypi"},
            {"column_num": 19, "ip_addr": "129.217.152.64", "mac_id": "b8:27:eb:36:da:22", "hostname": "raspberrypi"},
            {"column_num": 20, "ip_addr": "129.217.152.62", "mac_id": "b8:27:eb:f5:5d:04", "hostname": "raspberrypi"},
            {"column_num": 21, "ip_addr": "129.217.152.51", "mac_id": "b8:27:eb:88:8d:56", "hostname": "raspberrypi"},
            {"column_num": 22, "ip_addr": "129.217.152.87", "mac_id": "b8:27:eb:00:be:93", "hostname": "raspberrypi"},
            {"column_num": 23, "ip_addr": "129.217.152.33", "mac_id": "b8:27:eb:c0:10:ae", "hostname": "raspberrypi"},
            ]

#Import vicon_node_positions.csv
Vicon_Coords = pd.read_csv("vicon_node_positions.csv")
print(Vicon_Coords)


def convert_strip_id(mac_id):
    if mac_id == 'b8:27:eb:41:99:a0':
        return 1
    elif mac_id == 'b8:27:eb:c0:fd:6a':
        return 2
    elif mac_id == 'b8:27:eb:18:92:c7':
        return 3
    elif mac_id == 'b8:27:eb:53:f2:33':
        return 4
    elif mac_id == 'b8:27:eb:e7:6f:dc':
        return 5
    elif mac_id == 'b8:27:eb:38:4b:07':
        return 6
    elif mac_id == 'b8:27:eb:1b:cf:26':
        return 7
    elif mac_id == 'b8:27:eb:6d:0e:53':
        return 8
    elif mac_id == 'b8:27:eb:b7:a3:b7':
        return 9
    elif mac_id == 'b8:27:eb:be:dc:32':
        return 10
    elif mac_id == 'b8:27:eb:ff:a4:48':
        return 11
    elif mac_id == 'b8:27:eb:a9:7d:4d':
        return 12
    elif mac_id == 'b8:27:eb:c4:f8:c7':
        return 13
    elif mac_id == 'b8:27:eb:e4:43:6d':
        return 14
    elif mac_id == 'b8:27:eb:98:69:6e':
        return 15
    elif mac_id == 'b8:27:eb:75:c7:a2':
        return 16
    elif mac_id == 'b8:27:eb:09:3d:77':
        return 17
    elif mac_id == 'b8:27:eb:05:d8:4d':
        return 18
    elif mac_id == 'b8:27:eb:36:da:22':
        return 19
    elif mac_id == 'b8:27:eb:f5:5d:04':
        return 20
    elif mac_id == 'b8:27:eb:88:8d:56':
        return 21
    elif mac_id == 'b8:27:eb:00:be:93':
        return 22
    elif mac_id == 'b8:27:eb:c0:10:ae':
        return 23

#Convert unix timestamps to Julian date format
def Convert_to_julian_date(t_df):
    
    time_i = []
    offset = 2459828.75 #2459794.5 Julian epoch for 03.08.2022//Julian epoch for 5th August 2020: 2459067.00
    time_stamps = pd.DatetimeIndex(t_df['timestamp']).to_julian_date()
 
    for time_stamp in time_stamps:
        # time_stamp = ((time_stamp / 86400.0) + 2440587.5)
        time_i.append(time_stamp - offset)

    time_i_avg = np.mean([a for j,a in enumerate(time_i) if a>=0])
    t = time_i_avg * 24 * 60 * 60

    return t

def decode_data(j_msg, strip_id, node_id):
    
    global t_batch_i_old_arr, cnt
    
    delta_t = 0
    timestamp_i = 0
    data = j_msg['data']

    #check timestamp
    if t_batch_i_old_arr[strip_id-1][node_id-1] < j_msg['timestamp'] and t_batch_i_old_arr[strip_id-1][node_id-1] > 0:
    
        delta_t = (j_msg['timestamp'] - t_batch_i_old_arr[strip_id-1][node_id-1])/len(data)
        
        #print('1st: ', j_msg['timestamp'])
        #print('iter > 0: ', delta_t)
        t_i_batch_old = t_batch_i_old_arr[strip_id-1][node_id-1]
        t_batch_i_old_arr[strip_id-1][node_id-1] = j_msg['timestamp']
        print('iter > 0: ', t_i_batch_old)
        #print('2nd: ', t_batch_i_old_arr[strip_id-1][node_id-1])
        #cnt += 1
        #print('count: ', cnt)
        for i in range(len(data)):
            t_i = t_i_batch_old + ((1+i)*delta_t)
            t_i_array[strip_id-1][node_id-1] = t_i

            #print('iter > 0: ', t_i, t_i_array[strip_id-1][node_id-1], len(t_i_array))

            frame = ({'timestamp':t_i,'strip_id':strip_id,'node_id':node_id,
                'ax':data[i]['a'][0],'ay':data[i]['a'][1],'az':data[i]['a'][2],
                'gx':data[i]['g'][0],'gy':data[i]['g'][1],'gz':data[i]['g'][2],
                'mx':data[i]['m'][0],'my':data[i]['m'][1],'mz':data[i]['m'][2],
                'r':data[i]['r'][0]})
            for i, key in enumerate(KEYS):
                X_data[int(strip_id) - 1, node_id - 1, i] = frame[key]
        
    
            t_out = t_i_array.flatten()
            t_out_df = pd.DataFrame(t_out, columns=['timestamp'])
            t_out_df['timestamp'] = pd.to_datetime(t_out_df['timestamp'],unit='s')

            return t_out_df, X_data

        
    elif t_batch_i_old_arr[strip_id-1][node_id-1] == 0:
        #print('1st: ', j_msg['timestamp'])
        delta_t = 4/19
        
        t_i_batch_old = j_msg['timestamp'] - 4
        t_batch_i_old_arr[strip_id-1][node_id-1] = j_msg['timestamp']
        #print('2nd: ', t_i_batch_old, t_batch_i_old_arr[strip_id-1][node_id-1])
        print('iter 0: ', strip_id, node_id, t_i_batch_old, t_batch_i_old_arr[strip_id-1][node_id-1] )

        for i in range(len(data)):
            t_i = t_i_batch_old + ((1+i)*delta_t)
            t_i_array[strip_id-1][node_id-1] = t_i
            

            
            frame = ({'timestamp':t_i,'strip_id':strip_id,'node_id':node_id,
                'ax':data[i]['a'][0],'ay':data[i]['a'][1],'az':data[i]['a'][2],
                'gx':data[i]['g'][0],'gy':data[i]['g'][1],'gz':data[i]['g'][2],
                'mx':data[i]['m'][0],'my':data[i]['m'][1],'mz':data[i]['m'][2],
                'r':data[i]['r'][0]})
            
            for i, key in enumerate(KEYS):
                    #print(frame[key])
                X_data[int(strip_id) - 1, node_id - 1, i] = frame[key]
            
            t_out = t_i_array.flatten()
            #print(t_out)
            t_out_df = pd.DataFrame(t_out, columns=['timestamp'])
            t_out_df['timestamp'] = pd.to_datetime(t_out_df['timestamp'],unit='s')
                

                #print(t_i, t_out_df, X_data)
                #print(np.where(t_out_df['timestamp'].isna()))
            return t_out_df, X_data
    

# This is the Subscriber
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    for RPi in RPi_IPs:
        for node in range(1,16): # range 1 to 16 is an array from 1 to 15.
            client.subscribe("imu_reader/"+RPi['mac_id']+"/"+str(node))


def publish_msg(vicon_predict):

    x_coord = vicon_predict[0][0]
    y_coord = vicon_predict[0][1]

    print(x_coord, y_coord)

    # #construct topic for publishing message and received by unity laser program
    mqtt_publish_topic = 'imu_reader/viconpos'
    # publish_start_time = time()
    msg_to_laser = {"subject": "MR-1", "duration": 10, "color": "red", "shape": "circle",
                    "pointCount": 16, "animation": "pulse", "visible": "true",
                    "xpos": x_coord, "ypos": y_coord, "vicon_tracker": "false"}
    ret = client.publish(mqtt_publish_topic, json.dumps(msg_to_laser))
    print(msg_to_laser)

def on_message(client, userdata, msg):
    if msg.payload.decode():
        global count, count_plt
        j_msg = json.loads(msg.payload.decode('utf-8'))
        data = j_msg['data']

        strip_id = convert_strip_id(j_msg['strip_id'])
        node_id = int(j_msg['node_id'])

        #print(j_msg['timestamp'], j_msg['timestamp']-4)

        count += 1
        if count == 346:
            count_plt += 1
            print("test counter")
            t_df, X_input = decode_data(j_msg, strip_id, node_id)
            #print(np.where(t_df['timestamp'].isna()))
            #print('in loop: ',t_df, 'test X: ', X_input.shape, X_input)
            # t_input = getJulianFromUnix(timestamps_array)
            #coord_predict = model_wrapper.predict(X_input_clean, t_input_clean)
            # print("Vicon: ", coord_predict)
            count = 0
            # timestamps_array.clear()
            # X_input.clear()
        else:
            t_df, X_input = decode_data(j_msg, strip_id, node_id)
           # print(np.where(t_df['timestamp'].isna()))
            if count_plt > 1:
                t_input = Convert_to_julian_date(t_df)   
                X_input_predict = X_input.reshape([1, 23, 15, 10])
                t_input_predict = t_input.reshape([1,1])
                print('in loop: ',t_df, 'test X: ', X_input_predict.shape, t_input_predict.shape)
                coord_predict = model_wrapper.predict(X_input_predict, t_input_predict)
                print("vicon predict: ", coord_predict)


        #otsu library for thresholding

#fab imuread must run in parallel to trigger the broker
#Set paho mqtt callback
client = mqtt.Client("test_client") #create new instance
client.connect(broker_address, 8883, 60) #connect to broker
print("connecting to broker")

client.on_connect = on_connect
client.on_message = on_message
client.enable_bridge_mode()
client.loop_forever()

# try:
#     client.loop_forever()
# except:
#   print('disconnect')
#   client.disconnect()