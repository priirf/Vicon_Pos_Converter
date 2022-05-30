import paho.mqtt.client as mqtt #import the client1
import json
import numpy as np
import pandas as pd
from time import sleep, time

broker_address = "129.217.152.1"

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe('imu_reader/viconpos')

def on_message(client, userdata, msg):
  if msg.payload.decode():
      # print(msg.payload)
      j_msg = json.loads(msg.payload.decode('utf-8'))
      print(j_msg)

# set paho.mqtt callback
client = mqtt.Client()
client.connect("129.217.152.1",8883,60)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.loop_forever()
except KeyboardInterrupt:
    print('disconnect')
    client.disconnect()