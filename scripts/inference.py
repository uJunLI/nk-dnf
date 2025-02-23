import os
import time
import datetime
import yaml
import git
from copy import deepcopy
import jittor as jt
from jittor import nn
from jittor.dataset import Dataset

from utils import load_checkpoint, load_pretrained, save_image_jittor
from utils.config import parse_options, copy_cfg, ordered_dict_to_dict
from utils.metrics import get_psnr_jittor, get_ssim_jittor
from utils.logger import create_logger
from utils.AverageMeter import AverageMeter

from models import build_model
from datasets import build_test_loader
from forwards import build_forward

def main(config):
    jt.flags.use_cuda = 1
    data_loader = build_test_loader(config['data'], 2)

    logger.info(f"Creating model: {config['name']}/{config['model']['type']}")
    model = build_model(config['model'])
    logger.info(str(model))

    logger.info('Building forwards:')
    logger.info(f'Inference forward: {config["inference"]["forward_type"]}')
    forward = build_forward(config["inference"]["forward_type"])

    if config['train'].get('resume'):
        load_checkpoint(config, model, None, None, logger)

    if config['train'].get('pretrained') and (not config['train'].get('resume')):
        load_pretrained(config, model, logger)

    logger.info("Start Inference")
    start_time = time.time()
    inference(config, forward, model, data_loader, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Total time: {total_time_str}')


@jt.no_grad()
def inference(config, forward, model, data_loader, logger):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs, img_files = forward(config, model, data)
   
        output = outputs[config['inference']['which_stage']]
        output = jt.clamp(output, 0, 1) * 255

        result_path = os.path.join(config['output'], 'results', f'inference')
        os.makedirs(result_path, exist_ok=True)
        for i, result in enumerate(output):
            save_path = os.path.join(result_path, f'{os.path.basename(img_files[i])[:-4]}.png')
            save_image_jittor(result, save_path)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if config['testset_as_validset'] or idx % config['print_per_iter'] == 0 or idx == len(data_loader):
            logger.info(
                f'Infer: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\t'
                f'Data {data_time.sum:.4f} ({data_time.avg:.4f})\t')
               
    logger.info(f'Infer: Total Time: {datetime.timedelta(seconds=int(time.time()-start))}')


@jt.no_grad()
def test_metric_cuda(config, epoch, outputs, targets, image_paths, target_params=None):
    outputs = jt.clamp(outputs, 0, 1) * 255
    targets = jt.clamp(targets, 0, 1) * 255
    if config['test']['round']:
        outputs = outputs.round()
        targets = targets.round()
    psnr = get_psnr_jittor(outputs, targets)
    ssim = get_ssim_jittor(outputs, targets)

    if config['test']['save_image']:
        result_path = os.path.join(config['output'], 'results', f'test_{epoch:04d}')
        os.makedirs(result_path, exist_ok=True)
        save_path = os.path.join(result_path, f'{os.path.basename(image_paths[0])[:-4]}_{psnr.item():.2f}.png')
        save_image_jittor(outputs[0], save_path)

    return psnr, ssim



if __name__ == '__main__':
    args, config = parse_options()
    phase = 'infer'
    if 'inference' not in config:
        config['inference'] = deepcopy(config['test'])
    config['testset_as_validset'] = True
    config['eval_mode'] = True

    assert not args.auto_resume and \
           (args.resume is not None or config['train'].get('resume') is not None) or \
           (args.pretrain is not None or config['train'].get('pretrained') is not None)

    os.makedirs(config['output'], exist_ok=True)
    start_time = time.strftime("%y%m%d-%H%M", time.localtime())
    logger = create_logger(output_dir=config['output'], name=f"{config['tag']}", action=f"{phase}-{start_time}")
    path = os.path.join(config['output'], f"{phase}-{start_time}.yaml")

    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        if repo.is_dirty():
            logger.warning(f'Current work on commit: {sha}, however the repo is dirty (not committed)!')
        else:
            logger.info(f'Current work on commit: {sha}.')
    except git.exc.InvalidGitRepositoryError:
        logger.warning('No git repo base.')

    copy_cfg(config, path)
    logger.info(f"Full config saved to {path}")
    logger.info("Config:\n" + yaml.dump(ordered_dict_to_dict(config), default_flow_style=False, sort_keys=False))
    main(config)
