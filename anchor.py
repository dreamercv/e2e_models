import numpy as np
anchors = []
with open("/home/fb/project/models/Sparse4D-main/projects/e2e_models/keams_anchor.txt","r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(",")
        print(len(line))
        if len(line) != 11:continue
        anchor = list(map(float,line))
        anchors.append(anchor)
np.save("300clips_kmeans512_range_200.npy",np.array(anchors))