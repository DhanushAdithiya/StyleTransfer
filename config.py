import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10' : 'conv3_1',
    '19' : 'conv4_1',
    '21' : 'conv4_2',
    '28' : 'conv5_1',
}


STYLE_WEIGHTS = {
    'conv1_1' : 1.,
    'conv2_1' : 0.8,
    'conv3_1' : 0.5,
    'conv4_1' : 0.1
}

CONTENT_WEIGHT = 1
STYLE_WEIGHT = 1E6

SAVE_INTERVAL = 50
EPOCH = 2500
