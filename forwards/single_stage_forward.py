from jittor import Module, init, nn
from utils.registry import FORWARD_REGISTRY

@FORWARD_REGISTRY.register()
def ss_train_forward(config, model, data):
    raw = data['noisy_raw'].cuda()
    rgb_gt = data['clean_rgb'].cuda()

    rgb_out = model(raw)

    ###### | output          | label
    return {'rgb': rgb_out}, {'rgb': rgb_gt}

@FORWARD_REGISTRY.register()
def ss_test_forward(config, model, data):
    if not config['test'].get('cpu', False):
        raw = data['noisy_raw'].cuda()
        rgb_gt = data['clean_rgb'].cuda()
    else:
        raw = data['noisy_raw']
        rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']
    rgb_out = model(raw)
    return {'rgb': rgb_out}, {'rgb': rgb_gt}, img_files, lbl_files

@FORWARD_REGISTRY.register()
def ss_inference(config, model, data):
    raw = data['noisy_raw'].cuda()
    img_files = data['img_file']

    rgb_out = model(raw)

    ###### | output          | img names
    return {'rgb': rgb_out}, img_files
