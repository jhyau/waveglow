![WaveGlow](waveglow_logo.png "WaveGLow")

## WaveGlow: a Flow-based Generative Network for Speech Synthesis

### Ryan Prenger, Rafael Valle, and Bryan Catanzaro

In our recent [paper], we propose WaveGlow: a flow-based network capable of
generating high quality speech from mel-spectrograms. WaveGlow combines insights
from [Glow] and [WaveNet] in order to provide fast, efficient and high-quality
audio synthesis, without the need for auto-regression. WaveGlow is implemented
using only a single network, trained using only a single cost function:
maximizing the likelihood of the training data, which makes the training
procedure simple and stable.

Our [PyTorch] implementation produces audio samples at a rate of 1200 
kHz on an NVIDIA V100 GPU. Mean Opinion Scores show that it delivers audio
quality as good as the best publicly available WaveNet implementation.

Visit our [website] for audio samples.

## Setup

1. Clone our repo and initialize submodule

   ```command
   git clone https://github.com/NVIDIA/waveglow.git
   cd waveglow
   git submodule init
   git submodule update
   ```

2. Install requirements `pip3 install -r requirements.txt`

3. Install [Apex]

Note: when installing Apex for linux, if get error using the first command, try the python-only build command:

```
pip install -v --disable-pip-version-check --no-cache-dir ./
```

when following the quick start instructions in apex.

## Generate audio with our pre-existing model

1. Download our [published model]
2. Download [mel-spectrograms]
3. Generate audio `python3 inference.py -f <(ls mel_spectrograms/*.pt) -w waveglow_256channels.pt -o . --is_fp16 -s 0.6`  

N.b. use `convert_model.py` to convert your older models to the current model
with fused residual and skip connections.

## Train your own model

1. Download [LJ Speech Data]. In this example it's in `data/`

2. Make a list of the file names to use for training/testing

   ```command
   ls data/*.wav | tail -n+10 > train_files.txt
   ls data/*.wav | head -n10 > test_files.txt
   ```

Note that the above commands splits the list of .wav files in the directory such that `test_files.txt` has 10 files, and the rest are in `train_files.txt` 

3. Train your WaveGlow networks

   ```command
   mkdir checkpoints
   python train.py -c config.json
   ```

   For multi-GPU training replace `train.py` with `distributed.py`.  Only tested with single node and NCCL.

   For mixed precision training set `"fp16_run": true` on `config.json`.

4. Make test set mel-spectrograms

   `python mel2samp.py -f test_files.txt -o . -c config.json`

Note: For the output directory, don't include the final backslash (e.g. can have something like `-o data/asmr` but don't do something like `-o data/asmr/`)

5. Do inference with your network

   ```command
   ls *.pt > mel_files.txt
   python3 inference.py -f mel_files.txt -w checkpoints/waveglow_10000 -o . --is_fp16 -s 0.6
   ```

[//]: # (TODO)
[//]: # (PROVIDE INSTRUCTIONS FOR DOWNLOADING LJS)
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[paper]: https://arxiv.org/abs/1811.00002
[WaveNet implementation]: https://github.com/r9y9/wavenet_vocoder
[Glow]: https://blog.openai.com/glow/
[WaveNet]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[PyTorch]: http://pytorch.org
[published model]: https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF
[mel-spectrograms]: https://drive.google.com/file/d/1g_VXK2lpP9J25dQFhQwx7doWl_p20fXA/view?usp=sharing
[LJ Speech Data]: https://keithito.com/LJ-Speech-Dataset
[Apex]: https://github.com/nvidia/apex
