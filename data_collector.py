import paho.mqtt.client as mqtt #import the client1
import json
import numpy as np
import pandas as pd
from time import sleep, time
import seaborn as sns
import matplotlib.pylab as plt


broker_address = "129.217.152.1"
data = []

count = 0
ID = 1

rssi_mat = np.zeros((15,23))
data_mag = np.zeros((15,23))
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

# This is the Subscriber
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    for RPi in RPi_IPs:
        for node in range(1,16): # range 1 to 16 is an array from 1 to 15.
            client.subscribe("imu_reader/"+RPi['mac_id']+"/"+str(node))

def on_message(client, userdata, msg):
    if msg.payload.decode():
        global count, ID
        j_msg = json.loads(msg.payload.decode('utf-8'))
        data = j_msg['data']

        #print(type(j_msg['data']))
        strip_id = convert_strip_id(j_msg['strip_id'])
        j_msg['strip_id'] = convert_strip_id(j_msg['strip_id'])
        j_msg['strip_id'] = str(j_msg['strip_id'])

        #remove acceloremeter and gyroscope data
        for i in range(len(data)):
            del (data[i]['a'])
            del (data[i]['g'])

        # #construct new json message to be stored later
        # del (j_msg['data'])
        # rssi_avg = np.mean(rssi, axis=0)
        # magneto_avg = np.mean(magnetometer, axis=0)
        # avg_results = [{'r': str(rssi_avg), 'm': str(magneto_avg)}]
        # j_msg['data'] = avg_results

        # to plot the heatmap //uncomment this section
        count += 1
        if count == 346:
            #print(count)
            count = 0
            ID += 1
            print("--------------------------------------------------------")


        #attach the ID for each batch (345 nodes) of measurement
        j_msg['ID'] = ID
        print("json msg: ", j_msg)

        with open("sensor_floor_data.txt", "a+") as test_data:
            test_data.write(json.dumps(j_msg) + '\n')
        test_data.close()

#fab imuread must run in parallel to trigger the broker
#Set paho mqtt callback
client = mqtt.Client("test_client") #create new instance
client.connect(broker_address, 8883, 60) #connect to broker
print("connecting to broker")

client.on_connect = on_connect
client.on_message = on_message
#client.loop_forever()

try:
    client.loop_forever()
except:
  print('disconnect')
  client.disconnect()