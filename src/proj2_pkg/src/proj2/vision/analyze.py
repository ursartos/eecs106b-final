import cv2
import numpy as np

import random

WORDS = [w for w in open('/usr/share/dict/words').read().splitlines() if "'" not in w]

def dostuff(image):
    cv2.imshow('press SPACE to save the image', image)
    if cv2.waitKey(1) == ord(' '):
        name = 'images/' + random.choice(WORDS) + '-' + random.choice(WORDS) + '.jpg'
        cv2.imwrite(name, image)