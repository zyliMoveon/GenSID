import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.multiprocessing as mp
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

import argparse
from retry.api import retry_call

from Trainer import *
import torch.distributed as dist  # MODIFIED

def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed, wbrun):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 10 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = './models',
    name = 'Medium_v1',
    new = True,
    load_from = -1,
    image_size = 128,
    network_capacity = 8,
    fmap_max = 512,
    transparent = False,
    batch_size = 8,
    gradient_accumulate_every = 4,
    num_train_steps = 150000,
    learning_rate = 1e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers = 2,
    save_every = 3000,
    evaluate_every = 300000,
    generate = False,
    num_generate = 1,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 4,
    trunc_psi = 1,
    mixed_prob = 0.9,
    fp16 = True,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.5,
    aug_types = ['translation', 'cutout'],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dual_contrast_loss = False,
    dataset_aug_prob = 0.5,
    multi_gpus = True,
    calculate_fid_every = None,
    calculate_fid_num_images = 1782,
    clear_fid_cache = False,
    seed = 42,
    log = False
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = aug_types,
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        calculate_fid_num_images = calculate_fid_num_images,
        clear_fid_cache = clear_fid_cache,
        mixed_prob = mixed_prob,
        log = log
    )

    if generate:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    world_size = torch.cuda.device_count()

    if world_size == 1 or not multi_gpus:
        run_training(0, 1, model_args, data, load_from, new, num_train_steps, name, seed, wbrun)
        return

    mp.spawn(run_training,
        args=(world_size, model_args, data, load_from, new, num_train_steps, name, seed, wbrun),
        nprocs=world_size,
        join=True)

def main():
    parser = argparse.ArgumentParser(description='Train GenSID.')
    parser.add_argument('--data', type=str, default='./data/', help='Path to the data directory.')
    parser.add_argument('--results_dir', type=str, default='./results', help='Path to the results directory.')
    parser.add_argument('--models_dir', type=str, default='./models_dir/', help='Path to the models directory.')
    parser.add_argument('--name', type=str, default='default', help='Name of the experiment.')
    parser.add_argument('--new', action='store_true', help='Flag to train a new model.')
    parser.add_argument('--image_size', type=int, default=128, help='Size of the images.')
    parser.add_argument('--fmap_max', type=int, default=512, help='Maximum number of convolutional filter maps')
    parser.add_argument('--network_capacity', type=int, default=16, help='defines network capacity by setting lowest number of convolutional filters for the discriinator(4x) and generator(2x)')
    parser.add_argument('--trunc_psi', type=float, default=1.0, help='truncation parameter psi usually between 0 and 1')
    parser.add_argument('--dataset_aug_prob', type=float, default=0.5, help='defines augmentation probability for real images (horizontal flipping)')
    parser.add_argument('--aug-types', type=parse_list, default=["translation", "cutout"], help='Types of differentiable augmentations: selection from translation, cutout, contrast')
    parser.add_argument('--aug-prob', type=float, default=0.5, help='differentiable augmentation probability')
    parser.add_argument('--ttur_mult', type=float, default=1.5, help='Two Time-scale Update Rule multiplier')
    parser.add_argument('--save_every', type=int, default=3000, help='Save model checkpoint every n steps')
    parser.add_argument('--calculate_fid_every', type=int, default=None, help='Calculate FID, MD, AUC-PRD every n steps')
    parser.add_argument('--calculate_fid_num_images', type=int, default=500, help='calculate metrics using n generated volumes')
    parser.add_argument('--fp16', type=bool, default=True, help='mixed precision')
    parser.add_argument('--gradient-accumulate-every', type=int, default=1, help='gradient accumulation for n times in each step')
    parser.add_argument('--batch-size', type=int, default=6, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-06, help='learning rate of generator')
    parser.add_argument('--no_pl_reg', type=bool, default=False, help='whether to use path length regularization for the generator')
    parser.add_argument('--image-size', type=int, default=128, help='image size in width x heigth')

    args = parser.parse_args()

    kwargs = vars(args)

    train_from_folder(**kwargs)

if __name__ == "__main__":
    main()
