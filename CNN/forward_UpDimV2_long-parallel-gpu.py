# ------
# Modified by Audrey Monsimer
# November 2021
# ------

from pathlib import Path
import random
from collections import defaultdict

from PIL import Image
import torch
import torchvision as tv
import torchelie.recipes
import torchelie as tch
import torchelie.callbacks.callbacks as tcb
import argparse
import numpy as np
import soundfile as sf
import os
import scipy.signal as sg
from tqdm import tqdm, trange
from math import ceil


def main(args):
    batch_size = 64  #64 à la base
    num_feature = 4096
    num_classes = 10
    rng = np.random.RandomState(42)

    class UpDimV2(torch.nn.Module):

        def __init__(self, num_class):
            super(UpDimV2, self).__init__()
            self.activation = torch.nn.LeakyReLU(0.001, inplace=True)

            # Block 1D 1
            self.conv11 = torch.nn.Conv1d(1, 32, 3, 1, 1)
            self.norm11 = torch.nn.BatchNorm1d(32)
            self.conv21 = torch.nn.Conv1d(32, 32, 3, 2, 1)
            self.norm21 = torch.nn.BatchNorm1d(32)
            self.skip11 = torch.nn.Conv1d(1, 32, 1, 2)

            # Block 1D 2
            self.conv12 = torch.nn.Conv1d(32, 64, 3, 2, 1)
            self.norm12 = torch.nn.BatchNorm1d(64)
            self.conv22 = torch.nn.Conv1d(64, 128, 3, 2, 1)
            self.norm22 = torch.nn.BatchNorm1d(128)
            self.skip12 = torch.nn.Conv1d(32, 128, 1, 4)

            # Block 2D 1
            self.conv31 = torch.nn.Conv2d(1, 32, 3, 1, 1)
            self.norm31 = torch.nn.BatchNorm2d(32)
            self.conv41 = torch.nn.Conv2d(32, 32, 3, 2, 1)
            self.norm41 = torch.nn.BatchNorm2d(32)
            self.skip21 = torch.nn.Conv2d(1, 32, 1, 2)

            # Block 2D 2
            self.conv32 = torch.nn.Conv2d(32, 64, 3, 2, 1)
            self.norm32 = torch.nn.BatchNorm2d(64)
            self.conv42 = torch.nn.Conv2d(64, 128, 3, 2, 1)
            self.norm42 = torch.nn.BatchNorm2d(128)
            self.skip22 = torch.nn.Conv2d(32, 128, 1, 4)

            # Block 3D 1
            self.conv51 = torch.nn.Conv3d(1, 32, 3, (1, 2, 1), 1)
            self.norm51 = torch.nn.BatchNorm3d(32)
            self.conv61 = torch.nn.Conv3d(32, 64, 3, 2, 1)
            self.norm61 = torch.nn.BatchNorm3d(64)
            self.skip31 = torch.nn.Conv3d(1, 64, 1, (2, 4, 2))

            # Block 3D 2
            self.conv52 = torch.nn.Conv3d(64, 128, 3, 2, 1)
            self.norm52 = torch.nn.BatchNorm3d(128)
            self.conv62 = torch.nn.Conv3d(128, 256, 3, 2, 1)
            self.norm62 = torch.nn.BatchNorm3d(256)
            self.skip32 = torch.nn.Conv3d(64, 256, 1, 4)

            # Fully connected
            self.soft_max = torch.nn.Softmax(-1)  # If the time stride is too big, the softmax will be done on a singleton
            # which always ouput a 1
            self.fc1 = torch.nn.Linear(4096, 1024)
            self.fc2 = torch.nn.Linear(1024, 512)
            self.fc3 = torch.nn.Linear(512, num_class)

        def forward(self, x):
            # Block 1D 1
            out = self.conv11(x)
            out = self.norm11(out)
            out = self.activation(out)
            out = self.conv21(out)
            out = self.norm21(out)
            skip = self.skip11(x)
            out = self.activation(out + skip)

            # Block 1D 2
            skip = self.skip12(out)
            out = self.conv12(out)
            out = self.norm12(out)
            out = self.activation(out)
            out = self.conv22(out)
            out = self.norm22(out)
            out = self.activation(out + skip)

            # Block 2D 1
            out = out.reshape((lambda b, c, h: (b, 1, c, h))(*out.shape))
            skip = self.skip21(out)
            out = self.conv31(out)
            out = self.norm31(out)
            out = self.activation(out)
            out = self.conv41(out)
            out = self.norm41(out)
            out = self.activation(out + skip)

            # Block 2D 2
            skip = self.skip22(out)
            out = self.conv32(out)
            out = self.norm32(out)
            out = self.activation(out)
            out = self.conv42(out)
            out = self.norm42(out)
            out = self.activation(out + skip)

            # Block 3D 1
            out = out.reshape((lambda b, c, w, h: (b, 1, c, w, h))(*out.shape))
            skip = self.skip31(out)
            out = self.conv51(out)
            out = self.norm51(out)
            out = self.activation(out)
            out = self.conv61(out)
            out = self.norm61(out)
            out = self.activation(out + skip)

            # Block 3D 2
            skip = self.skip32(out)
            out = self.conv52(out)
            out = self.norm52(out)
            out = self.activation(out)
            out = self.conv62(out)
            out = self.norm62(out)
            out = self.activation(out + skip)

            # Fully connected
            out = torch.max(self.soft_max(out), -1)[0].reshape(-1, 4096)
            out = self.activation(self.fc1(out))
            out = self.activation(self.fc2(out))
            return self.fc3(out)


    model = torch.nn.DataParallel(UpDimV2(num_classes))
    model.load_state_dict((torch.load(args.weight)['model']))
    model.to('cuda')
    model.eval()

    if os.path.isfile(args.input_path):
        if args.input_path.endswith('.npy'):
            click_data = np.load(args.input_path)
            click_data = click_data[:, click_data.shape[1]//2 - num_feature//2:click_data.shape[1]//2+num_feature//2]
            with torch.no_grad():
                preds = np.empty((len(click_data), num_classes))
                for i in trange(len(click_data)//batch_size, desc=f'file: {args.input_path}'):
                    clicks = click_data[i*batch_size:(i+1)*batch_size]
                    clicks = torch.from_numpy(((clicks - clicks.mean(-1, keepdims=True))/(clicks.std(-1, keepdims=True) + 1e-18))[:, np.newaxis]).to('cuda').float()
                    preds[i*batch_size:(i+1)*batch_size] = model(clicks).cpu().numpy()
                if not (len(click_data) % batch_size):
                    clicks = click_data[-(len(click_data) % batch_size):]
                    clicks = torch.from_numpy(((clicks - clicks.mean(-1, keepdims=True))/(clicks.std(-1, keepdims=True) + 1e-18))[:, np.newaxis]).to('cuda').float()
                    preds[-(len(click_data) % batch_size):] = model(clicks).cpu().numpy()
            np.savetxt(args.output_path.rsplit('.',1)[0] + args.suffix, preds)    

        else:
            try :
                song, sr = sf.read(args.input_path, always_2d=True)
            
                song = song[:, args.channel]
                sos = sg.butter(3, 200_000/sr, 'lowpass', output='sos')
                song = sg.sosfiltfilt(sos, song)
                song = sg.resample(song, int(200_000/sr*len(song)))
                batch_pos = np.linspace(0, len(song) - num_feature, args.overlap * batch_size * ceil((len(song)//num_feature + 1)/batch_size)).astype(int)
                with torch.no_grad():
                    preds = np.empty((len(batch_pos)//batch_size, batch_size, num_classes))
                    for i, pos in enumerate(tqdm(batch_pos.reshape(-1, batch_size), desc=f'file: {args.input_path}')):
                        clicks = np.array([song[p:p+num_feature] for p in pos])
                        clicks = torch.from_numpy(((clicks - clicks.mean(-1, keepdims=True))/(clicks.std(-1, keepdims=True) + 1e-18))[:, np.newaxis]).to('cuda').float()
                        preds[i] = model(clicks).cpu().numpy()
                np.savetxt(args.output_path.rsplit('.',1)[0] + args.suffix, preds.reshape(-1, num_classes)) 
            except Exception as e:
                print(f'error with file {args.input_path}: {e}')
  
    else:
        for d, _, dire in os.walk(args.input_path):
            if args.output_path is not None:
                dout = os.path.join(args.output_path, d[len(args.input_path):])
                os.makedirs(dout, exist_ok=True)
            for f in tqdm(dire, desc=f'directory: {d}'):
                if f.rsplit('.', 1)[-1].lower() not in ['wav', 'mp3', 'ogg', 'flac']:
                    continue
                try:
                    current_file = os.path.join(d, f)
                    if args.output_path is None:
                        out_file = os.path.join(d, f).rsplit('.',1)[0] + args.suffix
                    else:
                        out_file = os.path.join(dout, f).rsplit('.',1)[0] + args.suffix
                    if os.path.isfile(out_file) and args.erase:
                        continue
                    if args.undersample is not None:
                        if np.random.random_sample() > args.undersample/100:
                            continue
                    song, sr = sf.read(current_file, always_2d=True)
                    song = song[:, args.channel]
                    sos = sg.butter(3, 200_000/sr, 'lowpass', output='sos')
                    song = sg.sosfiltfilt(sos, song)
                    song = sg.resample(song, int(200_000/sr*len(song)))
                    batch_pos = np.linspace(0, len(song) - num_feature, args.overlap * batch_size * ceil((len(song)//num_feature + 1)/batch_size)).astype(int)
                    with torch.no_grad():
                        preds = np.empty((len(batch_pos)//batch_size, batch_size, num_classes))
                        for i, pos in enumerate(tqdm(batch_pos.reshape(-1, batch_size), desc=f'file: {current_file}')):
                            clicks = np.array([song[p:p+num_feature] for p in pos])
                            clicks = torch.from_numpy(((clicks - clicks.mean(-1, keepdims=True))/(clicks.std(-1, keepdims=True) + 1e-18))[:, np.newaxis]).to('cuda').float()
                            preds[i] = model(clicks).cpu().numpy()
                    np.savetxt(out_file, preds.reshape(-1, num_classes))
                except Exception as e:
                    print(f'error with file {current_file}: {e}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Analyse wav(s) and return logits prediction. Use a softmax to have probabilities. The classes order is Gg, Gma, La, Mb, Me, Pm, Ssp, UDA, UDB, Zc")
    parser.add_argument("input_path", type=str, help="Folder or path")
    parser.add_argument("--weight", type=str, default='best_acc_updimv2_3dlong.pth', help="Model weight")
    parser.add_argument("--suffix", type=str, default='.pred', help="Suffix of the output file")
    parser.add_argument("--channel", type=int, default=0, help="Channel used for prediction")
    parser.add_argument("--overlap", type=int, default=2, help="Overlap factor of prediction windows (win_size/hop_size)")
    parser.add_argument("--undersample", type=float, default=None, help="In case of folders, only undersample percent of files will be computed")
    parser.add_argument("--output_path", type=str, help="Path to root dir of ouput. Only used if input is folder. Default to input_path")
    parser.add_argument("--erase", action='store_false', help="If out_file exist and erase not specified, file will be skip. (Only for folder input)")
    parser.add_argument("--first_file", type =int, default=None, help="First file in the file list. Use if input path is a file list in txt")
    parser.add_argument("--number_files", type =int, default=None, help="Number of files to run")
    parser.add_argument("--lot", type =str, default=None, help="LOT1, LOT2, LOTn ...")

    args = parser.parse_args()
    output_path_origin=args.output_path
    if args.input_path.endswith('.txt'):
        with open(args.input_path) as f:
            list_input = f.readlines()
        print(list_input[args.first_file:args.first_file+args.number_files])
        for path_temp in list_input[args.first_file:args.first_file+args.number_files]:
            path=path_temp.split('\n')[0]
            args.input_path=path
            print("arg input path = ",args.input_path)
            out_path=output_path_origin+args.input_path.split('/')[-2]+'/'+args.lot+'-'+args.input_path.split('/')[-2]+'-'+args.input_path.split('/')[-1]
            args.output_path=out_path
            print("arg output path = ",args.output_path)
            if not os.path.isfile(args.output_path.rsplit('.',1)[0] + args.suffix):
                os.makedirs(args.output_path.rsplit('/',1)[0], exist_ok=True)
                main(args)
            else :
                print('File ',args.output_path.rsplit('.',1)[0] + args.suffix,'already exist')
    else :
        main(args)



