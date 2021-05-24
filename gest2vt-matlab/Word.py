from Gesture import Gesture
import numpy as np

class Word:

    def __init__(self, gestures=[]):
        self.gestures = gestures
        self.length = 0
        self.index_file = 0
        self.index_word = None
        self.word = ""

    def read_gesture_file(self, filename):
        self.gestures = []
        with open(filename, 'r') as file:
            lines = file.readlines()
            self.word = lines[0]
            self.length = int(lines[1])
            self.index_file = int(lines[2])
            self.index_word = (int(lines[3]), int(lines[4]))
            for line in lines[5:]:
                if len(line) > 0 and line[0] != '#':
                    if line[-1] == '\n':
                        line = line[:-1]
                    args = line.split(',')
                    print(args)
                    gesture = Gesture(args[0], args[1], args[2], args[3], args[4])
                    self.gestures.append(gesture)
        self.gestures = np.array(self.gestures, dtype=object)

    def get_gestures(self):
        for i in np.nditer(self.gestures, flags=["refs_ok"], op_flags=["readwrite"]):
            yield i.item()
#         return self.gestures
