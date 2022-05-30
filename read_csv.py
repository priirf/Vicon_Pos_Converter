import pandas as pd

#Import vicon_node_positions.csv
Vicon_Coords = pd.read_csv("vicon_node_positions.csv")
Vicon_Coords = Vicon_Coords.astype({'strip_id' : 'string'})
Vicon_Coords = Vicon_Coords.astype({'node_id' : 'string'})
print(Vicon_Coords)

#df2 = Vicon_Coords.loc[(Vicon_Coords['strip_id'] == '1') & (Vicon_Coords['node_id'] == '1')]
df2 = Vicon_Coords.loc[(Vicon_Coords['strip_id'] == '1.0') & (Vicon_Coords['node_id'] == '1.0')]
print(df2['vicon_x'].values[0])
print(df2['vicon_y'].values[0])


