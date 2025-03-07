import torch
import os

from models.erase_model import EraseModel

from PIL import Image
from argparse import Namespace


opt = Namespace()
params = {
 'online': False,
 'saveOnline': False,
 'dataroot': './examples/poster',
 'checkpoints_dir': './checkpoints',
 'load_dir': None,
 'name': 'ste',
 'gpu_ids': [0],
 'model': 'erasenet',
 'input_nc': 3,
 'output_nc': 3,
 'ngf': 64,
 'ndf': 64,
 'netD': 'n_layers',
 'netG': 'resnet_9blocks',
 'n_layers_D': 5,
 'norm': 'instance',
 'init_type': 'normal',
 'init_gain': 0.02,
 'adg_start': True,
 'netD_M': False,
 'reward_type': '2',
 'no_dropout': False,
 'maskD': False,
 'mask_sigmoid': True,
 'PasteImage': False,
 'PasteText': False,
 'valid': 0,
 'domain_in': False,
 'dataset_mode': 'items',
 'gen_space': 'random',
 'serial_batches': True,
 'nThreads': 1,
 'batchSize': 1,
 'load_size': 512,
 'crop_size': 512,
 'gen_method': 'art',
 'max_dataset_size': float('inf'),
 'preprocess': 'resize',
 'flip': 0.0,
 'rotate': 0.3,
 'mask_mode': 1,
 'display_winsize': 256,
 'raw_mask_dilate': 4,
 'mask_dilate': 3,
 'seed': 66,
 'verbose': False,
 'which_epoch': 'best',
 'ntest': float('inf'),
 'D': False,
 'results_dir': '../results/',
 'aspect_ratio': 1.0,
 'phase': 'test',
 'how_many': float('inf'),
 'isTrain': False,
 'data_norm': False,
 'no_flip': True
}
for k, v in params.items():
    setattr(opt, k, v)

model = EraseModel()  
model.initialize(opt)
model.setup(opt)



from data.items_dataset import ItemsDataset
dataset = ItemsDataset()
dataset.initialize(opt)

dataset = torch.utils.data.DataLoader(
    dataset,
    batch_size = 1,
    shuffle = False,
    num_workers = 0
)


back_trans = dataset.back_transform if hasattr(dataset, "back_transform") else None

image_dir = "./rr"
print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals(back_trans)

    for label, image_numpy in visuals.items():
        image_name = f'test_{label}.png'
        save_path = os.path.join(image_dir, image_name)
        
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(save_path)

    img_path = model.get_image_paths()
    print('process image...%s, %s' % (i, img_path))
    break