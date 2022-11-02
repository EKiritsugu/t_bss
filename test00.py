print('hello world!')

import numpy as np

import soundfile as sf
from scipy import signal
import librosa
import pyroomacoustics as pra


import mir_eval.separation as mes

# import pandas as pd
# from pandas import DataFrame
import os
import sys

print('import done!')
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes

## 生成空间坐标
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes
class room_para:
    def __init__(self, n_sources):
        def mic_judge():
            center_room = self.wall / 2
            xy_delta = self.center_mic - center_room
            distance = np.sqrt(np.sum(xy_delta**2))
            if distance < 0.2:
                return False
            else:
                return True

        def source_judge():
            dis_mic = np.array([ np.sqrt(np.sum((self.loc_source[i] - self.center_mic)**2))
                       for i in range(n_sources)
                    ])
            dis_center = np.array([ np.sqrt(np.sum((self.loc_source[i] - self.wall / 2)**2))
                       for i in range(n_sources)
                    ])
            dis_source = np.array([ np.sqrt(np.sum((self.loc_source[i] - self.loc_source[j] )**2))
                           for i in range(n_sources)
                           for j in range(i+1, n_sources)])

            if np.min(dis_mic) > 1.5 and np.min(dis_center) > 0.2 and np.min(dis_source) > 1 :
                return True

        self.n_ch = n_sources
        self.T60 = 10 * random.randint(20, 60)
        self.fs = 16000

        self.wall = 5 + 5* np.array([random.random() , random.random()])
        self.upper = 3 + random.random()


        self.r_mic = 0.075 + 0.05 * random.random()
        self.z_mic = 1 + random.random()
        while True:
            self.center_mic = self.r_mic +  np.array([
                (self.wall[0] - 2* self.r_mic) * random.random() ,
                (self.wall[1] - 2* self.r_mic) * random.random()])
            if mic_judge():
                break
        self.theta_mic = (1 / n_sources) * np.arange(0 , n_sources) * 360 + random.randint(0, 360)
        self.loc_mic = self.r_mic * np.array( [
            np.cos(self.theta_mic * 2* np.pi / 360) ,
            np.sin(self.theta_mic * 2* np.pi / 360)]).T \
                       + self.center_mic

        self.z_source = 1.5 + 0.5 * np.random.rand(1, n_sources).squeeze()
        while True:
            self.loc_source = self.wall * np.random.rand(n_sources , 2)
            if source_judge():
                break

        self.xyz_source = np.hstack((self.loc_source, self.z_source[:, None]))
        self.xyz_mic = (np.hstack((self.loc_source, self.z_mic* np.ones((n_sources,1)))))
        self.xyz_box = np.hstack((self.wall, self.upper))


    def pic_show(self):
        fig,ax = plt.subplots()

        rect = mpathes.Rectangle([0,0], self.wall[0], self.wall[1], color = 'pink')
        ax.add_patch(rect)

        x_mic = self.loc_mic[:,0]
        y_mic = self.loc_mic[:,1]
        plt.scatter(x_mic, y_mic, c='blue', label = 'function')

        plt.scatter(self.loc_source[:,0], self.loc_source[:,1], c='green', label = 'function')


        plt.axis('equal')
        plt.grid()
        plt.show()

room = room_para(4)
print(room.xyz_box)
# print(room.xyz_mic)
# print(room.xyz_source)

room.pic_show()


import pyroomacoustics as pra


def simulate():
    box = room_para(4)

    rt60 = box.T60
    fs = box.fs
    n_ch = box.n_ch


    e_absorption , max_order = pra.inverse_sabine(rt60, box.xyz_box.tolist())
    echo_room = pra.ShoeBox(box.xyz_box, materials = pra.Material(e_absorption), max_order = max_order, fs = fs)

    for i in range(n_ch):

        echo_room.add_microphone_array(box.xyz_mic[i, :, None])
        echo_room.add_source(box.xyz_source[i, :, None])

    echo_room.compute_rir()


    fig, ax = echo_room.plot()
    ax.set_xlim([-1, 10])
    ax.set_ylim([-1, 10])
    ax.set_zlim([-1, 5])
    fig.show()

    # room.compute_rir()
    # ret = room.rir
    # return ret





simulate()
