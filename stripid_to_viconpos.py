import paho.mqtt.client as mqtt #import the client1
import json
import numpy as np
import pandas as pd
from time import sleep, time
import seaborn as sns
import matplotlib.pylab as plt


broker_address = "129.217.152.1"
data = []
rssi = []
rssi_avg = 0
magnetometer = []
magneto_avg = 0
count = 0
cond = True

rssi_mat = np.zeros((23,15))
data_mag = np.zeros((23,15))
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
        return 1.0
    if mac_id == 'b8:27:eb:c0:fd:6a':
        return 2.0
    if mac_id == 'b8:27:eb:18:92:c7':
        return 3.0
    if mac_id == 'b8:27:eb:53:f2:33':
        return 4.0
    if mac_id == 'b8:27:eb:e7:6f:dc':
        return 5.0
    if mac_id == 'b8:27:eb:38:4b:07':
        return 6.0
    if mac_id == 'b8:27:eb:1b:cf:26':
        return 7.0
    if mac_id == 'b8:27:eb:6d:0e:53':
        return 8.0
    if mac_id == 'b8:27:eb:b7:a3:b7':
        return 9.0
    if mac_id == 'b8:27:eb:be:dc:32':
        return 10.0
    elif mac_id == 'b8:27:eb:ff:a4:48':
        return 11.0
    elif mac_id == 'b8:27:eb:a9:7d:4d':
        return 12.0
    elif mac_id == 'b8:27:eb:c4:f8:c7':
        return 13.0
    elif mac_id == 'b8:27:eb:e4:43:6d':
        return 14.0
    elif mac_id == 'b8:27:eb:98:69:6e':
        return 15.0
    elif mac_id == 'b8:27:eb:75:c7:a2':
        return 16.0
    elif mac_id == 'b8:27:eb:09:3d:77':
        return 17.0
    elif mac_id == 'b8:27:eb:05:d8:4d':
        return 18.0
    elif mac_id == 'b8:27:eb:36:da:22':
        return 19.0
    elif mac_id == 'b8:27:eb:f5:5d:04':
        return 20.0
    elif mac_id == 'b8:27:eb:88:8d:56':
        return 21.0
    elif mac_id == 'b8:27:eb:00:be:93':
        return 22.0
    elif mac_id == 'b8:27:eb:c0:10:ae':
        return 23.0

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    for RPi in RPi_IPs:
        for node in range(1,16): # range 1 to 16 is an array from 1 to 15.
            client.subscribe("imu_reader/"+RPi['mac_id']+"/"+str(node))



def on_message(client, userdata, msg):
    if msg.payload.decode():
        global count
        j_msg = json.loads(msg.payload.decode('utf-8'))
        data = j_msg['data']

        #print(type(j_msg['data']))
        #print(j_msg)
        #strip_id = convert_strip_id(j_msg['strip_id'])
        #print(strip_id, j_msg['node_id'], j_msg['data'])
        for i in range(len(data)):
            rssi.append(data[i]['r'])
            #print((data[i]['r']))
            magnetometer.append(data[i]['m'])

        rssi_avg = np.mean(rssi, axis=0)
        magneto_avg = np.mean(magnetometer, axis=0)
        strip_id = convert_strip_id(j_msg['strip_id'])
        #to plot the heatmap //uncomment this section
        # count += 1
        # if count == 345:
        #     print(count)
        #     count = 0
        #     print("--------------------------------------------------------")
        #     print(rssi_mat)
        #     print(data_mag)
        #     sns.heatmap(rssi_mat, annot=True, cbar_kws={'label': 'RSSI'}, cmap="YlGnBu")
        #     plt.title("RSSI Heatmap")
        #     plt.xlabel("Node ID")
        #     plt.ylabel("Strip ID")
        #     plt.show()
        #     sns.heatmap(data_mag, annot=True, cbar_kws={'label': 'Magnetic Field'}, cmap="YlGnBu")
        #     plt.title("Magnetometer Heatmap")
        #     plt.xlabel("Node ID")
        #     plt.ylabel("Strip ID")
        #     plt.show()
        #     return cond == False
        #     #client.loop_stop()
        # else:
        #     rssi_mat[int(strip_id)-1][int(j_msg['node_id'])-1] = rssi_avg[0]
        #     data_mag[int(strip_id) - 1][int(j_msg['node_id']) - 1] = magneto_avg[0]
        #     print(count)

        timestamp = time()
        data_to_store = str(timestamp) + ", " + str(strip_id) + ", " + str(j_msg['node_id']) + ", " + str(rssi_avg[0]) + ", " + str(magneto_avg[0])


        with open("datalog.txt", "a") as test_data:
            test_data.write(data_to_store + '\n')
        test_data.close()

        print(strip_id, j_msg['node_id'], "rssi avg: ", rssi_avg,"avg magneto: ", magneto_avg)
        # #if rssi_avg < 0 & magneto_avg[0] > 0 & magneto_avg[1] > 0
        # static: rssi > -50; magneto =~ 60; surrounding: rssi =~ 56, magneto
        #if (rssi_avg > -80 and rssi_avg < -10 and magneto_avg[0] > 0 and magneto_avg[0] < 270) or (rssi_avg > -50 and rssi_avg < -5 and magneto_avg[0] > -80 and magneto_avg[0] < 0 ):
        #if False:
        #if magneto_avg[0] > 20:
        if strip_id > 3 and strip_id < 11 and rssi_avg > -67 and rssi_avg < -20 and magneto_avg[0] > -3 and magneto_avg[0] < 260:
            #print("Filtered: ", strip_id, j_msg['node_id'], rssi_avg, magneto_avg)
            # with open("datalog.txt", "a") as test_data:
            #     # test_data.write(json.dumps(data) + '\n')
            #     test_data.write("Filtered: " + data_to_store + '\n')
            # test_data.close()
            # print(strip_id, j_msg['node_id'], magneto_avg, magnetometer)
            df2 = Vicon_Coords.loc[(Vicon_Coords['strip_id'] == strip_id) & (Vicon_Coords['node_id'] == float(j_msg['node_id']))]
            x_coord = df2['vicon_x'].values[0].round(3)
            y_coord = df2['vicon_y'].values[0].round(3)

            print(x_coord, y_coord)

            # #construct topic for publishing
            mqtt_publish_topic = 'imu_reader/viconpos'
            publish_start_time = time()
            msg_to_laser = {"subject": "MR-1", "duration": 10, "color": "red", "shape": "circle", "pointCount": 16, "animation": "pulse", "visible": "true",
                            "xpos" : x_coord, "ypos" : y_coord, "vicon_tracker" : "false"}
                            # "target" : {x_coord, y_coord, 0.0}}
            ret = client.publish(mqtt_publish_topic, json.dumps(msg_to_laser))
            print(msg_to_laser)
        rssi.clear()
        magnetometer.clear()

#fab imuread must run in parallel to trigger the broker
#Set paho mqtt callback
client = mqtt.Client("test_client") #create new instance
client.connect(broker_address, 8883, 60) #connect to broker
print("connecting to broker")

client.on_connect = on_connect
client.on_message = on_message
client.enable_bridge_mode()
#client.loop_forever()
#client.loop_start()
#client.loop_forever()

# while True:
#     client.loop_forever()
#     if cond == False:
#         client.loop_stop()
#         break
try:
    while True:
        client.loop_forever()
        if cond == False:
            client.loop_stop()
            break
except:
  print('disconnect')
  client.disconnect()