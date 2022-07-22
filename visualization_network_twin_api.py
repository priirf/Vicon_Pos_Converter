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
count_plt = 0
cond = True

# rssi_mat = np.zeros((23,15))
# data_mag = np.zeros((23,15))

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

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    for RPi in RPi_IPs:
        for node in range(1,16): # range 1 to 16 is an array from 1 to 15.
            client.subscribe("imu_reader/"+RPi['mac_id']+"/"+str(node))

def store_and_publish_msg(strip_id, node_id, rssi, magnetometer):

    print("Filtered: ", strip_id, node_id, rssi, magnetometer)
    timestamp = time()
    data_to_store = str(timestamp) + ", " + str(strip_id) + ", " + str(node_id) + ", " + str(
        rssi) + ", " + str(magnetometer)

    with open("datalog.txt", "a") as test_data:
        test_data.write("Filtered: " + data_to_store + '\n')
    test_data.close()
    # print(strip_id, j_msg['node_id'], magneto_avg, magnetometer)

    df2 = Vicon_Coords.loc[
        (Vicon_Coords['strip_id'] == strip_id) & (Vicon_Coords['node_id'] == float(node_id))]
    x_coord = df2['vicon_x'].values[0].round(3)
    y_coord = df2['vicon_y'].values[0].round(3)

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
        count += 1
        if count == 346 and count_plt < 10:
            count_plt += 1
            #print(count)
            count = 0
            print("--------------------------------------------------------")
            print(rssi_mat)
            print(data_mag)
            plt.figure(figsize=(15, 9))
            #sns.set(font_scale=1.4)
            ax = sns.heatmap(rssi_mat, annot=True, cbar_kws={'label': 'RSSI'}, cmap="YlGnBu")
            ax.figure.axes[-1].yaxis.label.set_size(16)
            #sns.set(font_scale=1.4)
            plt.title("RSSI Heatmap", fontsize = 16)
            plt.ylabel("Node ID", fontsize = 16)
            plt.xlabel("Strip ID", fontsize = 16)
            #plt.figure(figsize=(1.589, 9.88), dpi=100)
            plt.tight_layout()
            plt.savefig('1907_RSSI_Static_test' + str(count_plt) + '.png')
            #plt.show()

            plt.figure(figsize=(15, 9))
            bx = sns.heatmap(data_mag, annot=True, cbar_kws={'label': 'Magnetic Field'}, cmap="YlGnBu")
            bx.figure.axes[-1].yaxis.label.set_size(16)
            #sns.set(font_scale=1.4)
            #sns.set(rc={"figure.figsize": (15, 9)})
            plt.title("Magnetometer Heatmap", fontsize = 16)
            plt.ylabel("Node ID", fontsize = 16)
            plt.xlabel("Strip ID", fontsize = 16)
            #plt.figure(figsize=(1.589, 9.88), dpi=100)
            plt.tight_layout()
            plt.savefig('1907_Mag_Static_test' + str(count_plt) + '.png')
            #plt.show()

        else:

            #store the rssi and magnetometer values to corresponding array index of strip and node ids
            # rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] = rssi_avg[0]
            # data_mag[int(strip_id) - 1][int(j_msg['node_id']) - 1] = magneto_avg[0]

            rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] = rssi_avg[0]
            data_mag[int(j_msg['node_id']) - 1][int(strip_id) - 1]  = magneto_avg[0]


            # conditions to determine the highest densed RSSI values in certain location
            if strip_id > 0 and strip_id < 22 and int(j_msg['node_id']) > 0 and int(j_msg['node_id']) < 14:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) + 1] < -20):

                if (rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] < -20 and
                    #rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) + 1][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id) - 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

            elif strip_id == 0 and int(j_msg['node_id']) > 0 and int(j_msg['node_id']) < 14:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] < -20):

                if (rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id) + 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

            elif strip_id == 22 and int(j_msg['node_id']) > 0 and int(j_msg['node_id']) < 14:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] < -20):

                if (rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id) - 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

            elif strip_id == 0 and int(j_msg['node_id']) == 0:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) + 1] < -20):

                if (rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id) + 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) + 1][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id) + 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

            elif strip_id == 0 and int(j_msg['node_id']) == 14:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) + 1][int(j_msg['node_id']) - 1] < -20):

                if (rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id) + 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) + 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

            elif strip_id == 22 and int(j_msg['node_id']) == 0:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) + 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) + 1] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) + 1] < -20):

                if (rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id) - 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) + 1][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id']) + 1][int(strip_id) - 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

            elif strip_id == 22 and int(j_msg['node_id']) == 14:

                # if (rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id']) - 1] < -20 and
                #     rssi_mat[int(strip_id)][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id)][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id'])] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id'])] < -20 and
                #     rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] > -70 and rssi_mat[int(strip_id) - 1][int(j_msg['node_id']) - 1] < -20):

                if (rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id)] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id)] < -20 and
                    rssi_mat[int(j_msg['node_id'])][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id'])][int(strip_id) - 1] < -20 and
                    rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] > -70 and rssi_mat[int(j_msg['node_id']) - 1][int(strip_id) - 1] < -20):

                        store_and_publish_msg(strip_id, j_msg['node_id'], rssi_avg[0], magneto_avg[0])

        timestamp = time()
        data_to_store = str(timestamp) + ", " + str(strip_id) + ", " + str(j_msg['node_id']) + ", " + str(rssi_avg[0]) + ", " + str(magneto_avg[0])

        with open("datalog.txt", "a") as test_data:
            test_data.write(data_to_store + '\n')
        test_data.close()

        #print(count)
        print(strip_id, j_msg['node_id'], "rssi avg: ", rssi_avg,"avg magneto: ", magneto_avg)

        rssi.clear()
        magnetometer.clear()

        #otsu library for thresholding

#fab imuread must run in parallel to trigger the broker
#Set paho mqtt callback
client = mqtt.Client("test_client") #create new instance
client.connect(broker_address, 8883, 60) #connect to broker
print("connecting to broker")

client.on_connect = on_connect
client.on_message = on_message
client.enable_bridge_mode()
#client.loop_forever()

try:
    client.loop_forever()
except:
  print('disconnect')
  client.disconnect()