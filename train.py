# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import argparse
import json
import os
import torch

#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from glow import WaveGlow, WaveGlowLoss
from mel2samp import Mel2Samp

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    #checkpoint_dict = torch.jit.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration'] # May need to set to zero if want to use pretrained ckpt to finetune
    optimizer.load_state_dict(checkpoint_dict['optimizer']) # May need to comment out if want to use pretrained checkpoint to finetune
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveGlow(**waveglow_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train(num_gpus, rank, group_name, root_path, audio_file_suffix, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, valid_batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard, wandb_project):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    criterion = WaveGlowLoss(sigma)
    model = WaveGlow(**waveglow_config).cuda()

    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    #=====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = Mel2Samp(**data_config, root_path=root_path, file_suffix=audio_file_suffix)
    validset = Mel2Samp(train=False, root_path=root_path, file_suffix=audio_file_suffix, **data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    valid_sampler = DistributedSampler(validset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    valid_loader = DataLoader(validset, num_workers=1, shuffle=False,
                              sampler=valid_sampler,
                              batch_size=valid_batch_size,
                              pin_memory=False,
                              drop_last=True)
    valid_iterator = iter(valid_loader)

    # Get shared output_directory ready
    if rank == 0:
        import wandb
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)
        os.environ['WANDB_DIR'] = os.path.abspath(output_directory)
        wandb.init(project=wandb_project)
        wandb.config.learning_rate = learning_rate
        wandb.config.sigma = sigma
        wandb.config.batch_size = batch_size
        wandb.config.seed = seed
        wandb.fp16_run = fp16_run
        wandb.data_config = data_config
        wandb.waveglow_config = waveglow_config

    if with_tensorboard and rank == 0:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    since_checkpoint = 0
    since_valid_check = 0
    valid_check_frequency = 4
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            mel, audio = batch
            mel = torch.autograd.Variable(mel.cuda())
            audio = torch.autograd.Variable(audio.cuda())
            outputs = model((mel, audio))

            loss = criterion(outputs)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Run evaluation on validation set
            if since_valid_check % valid_check_frequency == 0:
                try:
                    valid_batch = next(valid_iterator)
                except StopIteration:
                    valid_iterator = iter(valid_loader)
                    valid_batch = next(valid_iterator)
                valid_mel, valid_audio = valid_batch
                valid_mel = torch.autograd.Variable(valid_mel.cuda())
                valid_audio = torch.autograd.Variable(valid_audio.cuda())
                with torch.no_grad():
                    valid_outputs = model((valid_mel, valid_audio))
                    valid_loss = criterion(valid_outputs)
                    if num_gpus > 1:
                        reduced_valid_loss = reduce_tensor(valid_loss.data, num_gpus).item()
                    else:
                        reduced_valid_loss = valid_loss.item()
                if rank == 0:
                    wandb.log({'valid_loss': reduced_valid_loss}, commit=False, step=iteration)
                    print("{}:\t{:.9f},\t{:.9f}".format(iteration, reduced_loss, reduced_valid_loss))
            else:
                print("{}:\t{:.9f}".format(iteration, reduced_loss))

            #print("{}:\t{:.9f}".format(iteration, reduced_loss))
            optimizer.step()
            
            if rank == 0:
                wandb.log({'train_loss': reduced_loss}, step=iteration)
                if with_tensorboard:
                    logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)


            #if with_tensorboard and rank == 0:
            #    logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)

            since_checkpoint += num_gpus
            since_valid_check += 1

            if since_checkpoint >= iters_per_checkpoint or iteration == 0:
                if rank == 0:
                    checkpoint_path = "{}/waveglow_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)
                since_checkpoint = 0


            #if (iteration % iters_per_checkpoint == 0):
            #    if rank == 0:
            #        checkpoint_path = "{}/waveglow_{}".format(
            #            output_directory, iteration)
            #        save_checkpoint(model, optimizer, learning_rate, iteration,
            #                        checkpoint_path)

            #iteration += 1
            iteration += num_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    parser.add_argument('--root_path', default=None, type=str, help="root path of audio")
    parser.add_argument('--audio_suffix', default=None, type=str, help="suffix of audio wave files")
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, args.root_path, args.audio_suffix, **train_config)
