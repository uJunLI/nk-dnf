from utils.registry import FORWARD_REGISTRY
import jittor as jt
import logging


@FORWARD_REGISTRY.register(suffix='DNF')
def train_forward(config, model, data):
    raw = jt.array(data['noisy_raw'])  # Use jittor array for tensor
    raw_gt = jt.array(data['clean_raw'])
    rgb_gt = jt.array(data['clean_rgb'])
    
    rgb_out, raw_out = model(raw)
    ###### | output                          | label
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}


@FORWARD_REGISTRY.register(suffix='DNF')
def test_forward(config, model, data):
    if not config['test'].get('cpu', False):
        raw = jt.array(data['noisy_raw'])  # Using jittor tensors
        raw_gt = jt.array(data['clean_raw'])
        rgb_gt = jt.array(data['clean_rgb'])
    else:
        raw = jt.array(data['noisy_raw'])  # Or handle as needed
        raw_gt = jt.array(data['clean_raw'])
        rgb_gt = jt.array(data['clean_rgb'])

    img_files = data['img_file']
    lbl_files = data['lbl_file']

    rgb_out, raw_out = model(raw)

    ###### | output                          | label                         | img and label names
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register(suffix='DNF')  # without label, for inference only
def inference(config, model, data):
    raw = jt.array(data['noisy_raw'])  # Using jittor arrays
    img_files = data['img_file']

    rgb_out, raw_out = model(raw)

    ###### | output                          | img names
    return {'rgb': rgb_out, 'raw': raw_out}, img_files


# @FORWARD_REGISTRY.register()
# def DNF_profile(config, model, data, logger):
#     x = data['noisy_raw'].cuda()
#     flops = FlopCountAnalysis(model, x)
#     logger.info('Detaild FLOPs:\n' + flop_count_table(flops))
#     flops_total = flops.total()
#     logger.info(f"Total FLOPs: {flops_total:,}")


@FORWARD_REGISTRY.register()
def DNF_profile(config, model, data, logger):
    x = jt.array(data['noisy_raw'])  # Using jittor tensor
    # Here you would implement your own FLOP counting or use an external library for Jittor
    # For now, we'll log the shape of the tensor as a placeholder for FLOP counting
    logger.info(f"Input tensor shape: {x.shape}")
    # TODO: Implement or integrate FLOP counting for Jittor
    # For simplicity, just log the number of elements
    logger.info(f"Number of elements in input tensor: {x.numel()}")