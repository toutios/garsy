import sys
import Tools as tl
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as text
import yaml
from anytree.importer import DictImporter
from Gesture import Gesture
from Word import Word
from pprint import pprint  # just for nice printing
from anytree import RenderTree  # just for nice printing

word_text_list = [
    # "1_beet_gestures_opt_ma.txt",
    # "2_bit_gestures_opt_ma.txt",
    # "4_bet_gestures_opt_ma.txt",
    # "5_bat_gestures_opt_ma.txt",
    # "6_pot_gestures_opt_ma.txt",
    # "7_but_gestures_opt_ma.txt",
    # "8_bought_gestures_opt_ma.txt",
    # "9_boat_gestures_opt_ma.txt",
    # "10_boot_gestures_opt_ma.txt",
    # "11_put_gestures_opt_ma.txt"
    # "bide_opt.txt",
    "span_M.txt"
]

for file_name in word_text_list:

    word = Word()
    word.read_gesture_file("../reference/Words/Paper/" + file_name)

    N = word.length

    box_height = 10

    constriction_labels = ["Bilabial", "Alveolar", "Palatal", "Velar", "Pharyngeal", "Velopharyngeal"]

    figure, axis = plt.subplots(figsize=(10,5))

    for i in range(len(constriction_labels)):
        axis.annotate(constriction_labels[i], (-100, i + 0.5), color='b',
                      weight='bold', fontsize=6, ha='center', va='center')

    for gesture in word.get_gestures():
        rect = patches.Rectangle((int(gesture.start_s * 1000), gesture.mouth_part), int((gesture.end_s - gesture.start_s) * 1000), 1)
        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.0
        cy = ry + rect.get_height() / 2.0
        axis.annotate("(" + str(gesture.stiffness) + ", " + str(round(gesture.degree * 2.4, 2)) + ")", (cx, cy), color='w',
                      weight='bold', fontsize=6, ha='center', va='center')
        axis.add_artist(rect)

    axis.set_xlim((-200, N))
    axis.set_ylim((0, 6))
    axis.get_yaxis().set_visible(False)
    # plt.show()
    plt.savefig('../reference/Images/' + file_name.split("_")[0] + "_" + file_name.split("_")[1] + '_opt.png')
