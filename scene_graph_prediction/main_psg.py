import warnings
from collections import Counter

warnings.filterwarnings('ignore')

import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers_psg.dataset.psg_dataset import PSGDataset
from scene_graph_prediction.scene_graph_helpers_psg.model.scene_graph_prediction_model_psg import SGPNModelWrapperPSG
from pytorch_lightning.callbacks import ModelCheckpoint
from torchinfo import summary


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers_psg/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def find_checkpoint_path(log_dir: str):
    def epoch_int(file_str):
        return int(file_str.split('=')[1].replace('.ckpt', ''))

    log_dir = Path(log_dir)
    checkpoint_folder = log_dir / 'checkpoints'
    checkpoints = sorted(checkpoint_folder.glob('*.ckpt'), key=lambda x: epoch_int(x.name), reverse=True)
    if len(checkpoints) == 0:
        return None
    return checkpoints[0]


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    args = parser.parse_args()
    config = config_loader(args.config)
    pl.seed_everything(config['SEED'], workers=True)
    mode = 'train'
    # mode = 'evaluate'

    name = args.config.replace('.json', '')
    print(f'Running {name}')

    logger = pl.loggers.TensorBoardLogger('scene_graph_prediction/scene_graph_helpers_psg/logs', name=name, version=0)
    checkpoint_path = find_checkpoint_path(logger.log_dir)
    token_counter = Counter()
    if mode == 'train':
        train_dataset = PSGDataset(config, 'train')
        val_dataset = PSGDataset(config, 'test')
        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                collate_fn=val_dataset.collate_fn)
        model = SGPNModelWrapperPSG(config, ind_to_classes=train_dataset.ind_to_classes, ind_to_predicates=train_dataset.ind_to_predicates)
        checkpoint = ModelCheckpoint(filename='{epoch}', save_top_k=-1, every_n_epochs=1)
        trainer = pl.Trainer(devices=1, strategy='ddp',  # TODO, if distributed causes issues, num_workers=0 and or pin_memory=False
                             max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=2, log_every_n_steps=50,
                             num_sanity_val_steps=0,
                             callbacks=[checkpoint, pl.callbacks.progress.RichProgressBar()], benchmark=False,
                             precision=16,
                             gradient_clip_val=config['GRAD_CLIP'], detect_anomaly=False,
                             accumulate_grad_batches=config['N_GRAD_ACCUM'])

        # count the frequency of each token in the training set, and save it as json
        # for batch in tqdm(train_loader,desc='Counting tokens'):
        #     for elem in batch[1]:
        #         token_counter.update(elem['relation'][:,2].tolist())
        #         # token_counter.update(elem['relation'].flatten().tolist())
        # with open('scene_graph_prediction/scene_graph_helpers_psg/dataset/pix2sg_pred_frequencies.json', 'w') as f:
        #     json.dump({k:v for k,v in sorted(token_counter.items())}, f)
        # # with open('scene_graph_prediction/scene_graph_helpers_psg/dataset/pix2sg_token_frequencies.json', 'w') as f:
        # #     json.dump({k:v for k,v in sorted(token_counter.items())}, f)

        print('Start Training')
        summary(model, depth=5)
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_path)

    elif mode == 'evaluate':
        train_dataset = PSGDataset(config, 'train')
        eval_dataset = PSGDataset(config, 'test')
        # print size of test set
        print(f'Test set size: {len(eval_dataset)}')
        eval_loader = DataLoader(eval_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True,
                                 collate_fn=eval_dataset.collate_fn)
        checkpoint_path = 'scene_graph_prediction/scene_graph_helpers_psg/logs/pix2sg_b5/version_0/checkpoints/epoch=129.ckpt'
        # checkpoint_path = 'scene_graph_prediction/scene_graph_helpers_psg/logs/pix2sg_b7/version_0/checkpoints/epoch=129.ckpt'
        # checkpoint_path = 'scene_graph_prediction/scene_graph_helpers_psg/logs/pix2sg_eva_base_freeze8/version_0/checkpoints/epoch=129.ckpt'
        # checkpoint_path = 'scene_graph_prediction/scene_graph_helpers_psg/logs/pix2sg_eva_large_freeze16/version_0/checkpoints/epoch=129.ckpt'
        model = SGPNModelWrapperPSG.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, ind_to_classes=train_dataset.ind_to_classes,
                                                         ind_to_predicates=train_dataset.ind_to_predicates)

        trainer = pl.Trainer(gpus=1, max_epochs=config['MAX_EPOCHES'], logger=logger, check_val_every_n_epoch=1, log_every_n_steps=50, num_sanity_val_steps=0,
                             callbacks=[pl.callbacks.progress.RichProgressBar()])
        trainer.validate(model, eval_loader, ckpt_path=checkpoint_path)


if __name__ == '__main__':
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
