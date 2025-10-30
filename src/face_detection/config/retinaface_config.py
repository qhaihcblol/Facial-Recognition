def get_config(network):
    configs = {
        "mobilenet_v2": cfg_mnet_v2,
    }
    return configs.get(network, None)

cfg_mnet_v2 = {
    'name': 'mobilenet_v2_1.0',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 16, #16 #32
    'epochs': 250,
    'milestones': [190, 220],
    'image_size': 640,
    'pretrain': True,
    'return_layers': [6, 13, 18],
    'in_channel': 32,
    'out_channel': 128
}