from sys import argv

# display constants
NO_DISPLAY = '-nodisplay' in argv
LAYER_DIM = (300, 300)

# extraction constants
SAMPLE_RATE = 44100
SIZE = 2206
FFT_SIZE = SIZE//2
WINDOW = (1, FFT_SIZE)

# training constants
# iterations for ANN training
TRAINING_ITERATIONS = 200
TERMINATION_VALUE = 0
VOLUME_THRESHOLD = 0.01
NOTES = (10, 60)
# cross validation parts
PARTS = 2

NAMES = [\
            'taylor',\
            'agnes',\
            'robert',\
            'darian',\
            'nat',\
            ]
