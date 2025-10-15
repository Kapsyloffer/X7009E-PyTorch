import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", type=str, help="Path to the CSV file")
args = parser.parse_args()

data = {}  # This will store {ID: [station_times]}

with open(args.csv_file, "r", newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        obj_id = row[1]              # Second column, e.g., 130
        station_time = float(row[3]) # Float value column
        if obj_id not in data:
            data[obj_id] = []
        data[obj_id].append(station_time)

stations = 30
row_height = 1
obj_spacing = 420  

fig, ax = plt.subplots(figsize=(20, 12))  

colors = ["skyblue", "salmon", "gold", "lightgreen"]

slot_width = 75
max_x = len(data) * obj_spacing


overlap_count = 0
placed_rects = {y: [] for y in range(stations)}  
overlap_boxes = []  


for obj_idx, (obj_id, station_times) in enumerate(data.items()):
    color = colors[obj_idx % len(colors)]
    start_x = (obj_idx) * obj_spacing 

    for station_idx, value in enumerate(station_times):
        value = value * 0.6 #Cm => seconds
        y = station_idx  # 0-based row index
        spacing = 0 # (obj_spacing - value - slot_width) / 2
        x_start = start_x + (station_idx) * obj_spacing + spacing
        x_end = x_start + value 

        ax.add_patch(patches.Rectangle(
            (x_start, y),
            value,
            row_height,
            facecolor=color,
            edgecolor='black'
        ))

for x, y, width, height in overlap_boxes:
    ax.add_patch(patches.Rectangle(
        (x, y),
        width,
        height,
        facecolor='red',
        alpha=0.6,
        edgecolor='black'
    ))

tick_interval = obj_spacing
ax.set_xticks(range(0, max_x + tick_interval, tick_interval))
ax.set_xticklabels([
    str(int(i / obj_spacing)) if (i // obj_spacing) % 5 == 0 else "" 
    for i in range(0, max_x + tick_interval, tick_interval)
])
ax.set_xlabel("Time")

ax.set_yticks(range(stations))
ax.set_yticklabels([f"S{i+1}" for i in range(stations)])
ax.set_ylabel("Stations")

ax.set_title(f"Assembly Line Visualization (Overlaps: {overlap_count})")
ax.invert_yaxis()
for i in range(len(data) + stations):
    x = i * obj_spacing
    ax.add_patch(patches.Rectangle(
        (x - slot_width, 0),
        2 * slot_width,
        stations,
        # facecolor='gray',
        alpha=0.5,
        edgecolor='white'
    ))
    ax.axvline(x=x, color='black', linewidth=1)


plt.show()

print(f"Total overlaps detected: {overlap_count}")

