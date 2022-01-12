# imports
import os
import pathlib
import json
from dotenv import load_dotenv

import albumentations as A
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from torch.utils.data import DataLoader
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from custom_dataset import ObjectDetectionDataSet
from detection.faster_RCNN import FasterRCNN_lightning, get_fasterRCNN_resnet
from detection.transformations import Clip, ComposeDouble, AlbumentationWrapper
from detection.transformations import FunctionWrapperDouble, normalize_01
from detection.utils import collate_double, stats_dataset
from detection.utils import log_mapping_neptune, log_model_neptune, log_packages_neptune

# hyper-parameters
params = {'BATCH_SIZE': 2,
          'OWNER': 'feid',  # your username in neptune
          'SAVE_DIR': None,  # checkpoints will be saved to cwd
          'LOG_MODEL': False,  # whether to log the model to neptune after training
          'GPU': 1,  # set to None for cpu training
          'LR': 0.001,
          'PRECISION': 32,
          'CLASSES': 2,
          'SEED': 42,
          'PROJECT': 'Balloon',
          'EXPERIMENT': 'balloon',
          'MAXEPOCHS': 100,
          'PATIENCE': 50,
          'BACKBONE': 'resnet34',
          'FPN': False,
          'ANCHOR_SIZE': ((32, 64, 128, 256, 512),),
          'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
          'MIN_SIZE': 1024,
          'MAX_SIZE': 1024,
          'IMG_MEAN': [0.485, 0.456, 0.406],
          'IMG_STD': [0.229, 0.224, 0.225],
          'IOU_THRESHOLD': 0.5
          }


def main():
    # api key, https://github.com/neptune-ai/neptune-client
    load_dotenv() # read environment variables
    api_key = os.environ['NEPTUNE']  # if this throws an error, you didn't set your env var

    # save directory
    save_dir = os.getcwd() if not params['SAVE_DIR'] else params['SAVE_DIR']

    # custom dataset directory
    data_path = 'dataset/balloon'
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')
    
    # label mapping, starting at 1, as the background is assigned 0
    mapping = {
        'balloon': 1,
    }

    # training transformations and augmentations
    transforms_training = ComposeDouble([
        Clip(),
        AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
        AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
        #AlbumentationWrapper(albumentation=A.VerticalFlip(p=0.5)),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    # validation transformations
    transforms_validation = ComposeDouble([
        Clip(),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])


    # random seed
    seed_everything(params['SEED'])

    # dataset training
    dataset_train = ObjectDetectionDataSet(data_path=train_path,
                                           transform=transforms_training,
                                           mapping=mapping)

    # dataset validation
    dataset_valid = ObjectDetectionDataSet(data_path=val_path,
                                           transform=transforms_validation,
                                           mapping=mapping)

    # dataloader training
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=params['BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=collate_double)

    # dataloader validation
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate_double)
    
    # Datasets statistics exploration
    if False:
        stats_train = stats_dataset(dataset_train)
        transform = GeneralizedRCNNTransform(min_size=1024,
                                         max_size=1024,
                                         image_mean=[0.485, 0.456, 0.406],
                                         image_std=[0.229, 0.224, 0.225])
        stats_train_transform = stats_dataset(dataset_train, transform)
        print(stats_train)
        print(stats_train_transform)

    # neptune logger
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        project_name=f'{params["OWNER"]}/{params["PROJECT"]}',  # use your neptune name here
        experiment_name=params['EXPERIMENT'],
        params=params
    )

    assert neptune_logger.name  # http GET request to check if the project exists

    # model init
    model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                                  backbone_name=params['BACKBONE'],
                                  anchor_size=params['ANCHOR_SIZE'],
                                  aspect_ratios=params['ASPECT_RATIOS'],
                                  fpn=params['FPN'],
                                  min_size=params['MIN_SIZE'],
                                  max_size=params['MAX_SIZE'])

    # lightning init
    task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])

    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
    learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
    early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=params['PATIENCE'], mode='max')

    # trainer init
    trainer = Trainer(gpus=params['GPU'],
                      precision=params['PRECISION'],  # try 16 with enable_pl_optimizer=False
                      callbacks=[checkpoint_callback, learningrate_callback, early_stopping_callback],
                      default_root_dir=save_dir,  # where checkpoints are saved to
                      logger=neptune_logger,
                      log_every_n_steps=1,
                      num_sanity_val_steps=0,
                      )

    # start training
    trainer.max_epochs = params['MAXEPOCHS']
    trainer.fit(task,
                train_dataloader=dataloader_train,
                val_dataloaders=dataloader_valid)

    # start testing
    #trainer.test(ckpt_path='best', test_dataloaders=dataloader_valid)

    # log packages
    log_packages_neptune(neptune_logger)

    # log mapping as table
    log_mapping_neptune(mapping, neptune_logger)

    # log model
    if params['LOG_MODEL']:
        checkpoint_path = pathlib.Path(checkpoint_callback.best_model_path)
        log_model_neptune(checkpoint_path=checkpoint_path,
                          save_directory=pathlib.Path.home(),
                          name='best_model.pt',
                          neptune_logger=neptune_logger)

    # stop logger
    neptune_logger.experiment.stop()
    print('Finished')


if __name__ == '__main__':
    main()
