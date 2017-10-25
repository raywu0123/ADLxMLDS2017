import numpy as np
from tqdm import tqdm
import argparse
from os.path import join

parser = argparse.ArgumentParser(description='parse ark (and lab) file to tfr.')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='./data')
args = parser.parse_args()

def get_intchar_map(path):
  phone2int = {}
  phone2char = {}
  with open(path) as file:
    for line in file:
      sep = line[:-1].split('\t')
      phone2int[sep[0]] = int(sep[1])
      phone2char[sep[0]] = str(sep[2])
  return phone2int, phone2char

phone2int, phone2char = get_intchar_map(join(args.data_dir,'48phone_char.map'))

def get_label(path):
  ark = {}
  pre_id = ''
  with open(path) as file:
    single_id_ark = []
    id = ''
    for line in file:
      line = line.strip('\n').split(',')
      id = '_'.join(line[0].split('_')[:2])
      if (id != pre_id and len(single_id_ark) != 0):
        ark[pre_id] = single_id_ark
        single_id_ark = []
      pre_id = id
      single_id_ark.append(phone2int[line[1]])
    ark[id] = single_id_ark
  return ark

def read_ark(path):
  ark = {}
  pre_id = ''
  keys = []
  with open(path) as file:
    single_id_ark = []
    id = ''
    for line in file:
      line = line.strip('\n').split(' ')
      id = '_'.join(line[0].split('_')[:2])
      if (id != pre_id and len(single_id_ark) != 0):
        ark[pre_id] = single_id_ark
        keys.append(pre_id)
        single_id_ark = []
      pre_id = id
      single_id_ark.append(np.array(line[1:], dtype=float))
    ark[id] = single_id_ark
    keys.append(pre_id)
  return ark, keys

def write_npy(ark_path, output_path, label_path=None):
  sequences, keys = read_ark(ark_path)
  label_sequences = []
  if args.mode == 'train':
    label_sequences = get_label(label_path)
    assert (sequences.keys() == label_sequences.keys())

  print('Finish reading arks and labels.')
  all_frames = []
  all_labels = []
  for key in tqdm(keys):
    # print(key)
    for frame_id in range(len(sequences[key])):
      all_frames.append(sequences[key][frame_id])
      if args.mode == 'train':  all_labels.append(label_sequences[key][frame_id])

  if args.mode == 'train':
    assert (len(all_frames) == len(all_labels))
  print('num of frames = ', len(all_frames))


  def write_file(filename, nparray):
    with open(join(output_path, filename), 'wb+') as file:
      np.save(file, nparray)
      print('Wrote {} frames to {}'.format(nparray.shape[0], file.name))
  if args.mode == 'train':
    write_file('labels.npy', np.array(all_labels, dtype=int))
    write_file('trainframes.npy', np.array(all_frames, dtype=float))
  elif args.mode =='test':
    write_file('testframes.npy', np.array(all_frames, dtype=float))


if args.mode == 'train':
  ark_path = join(args.data_dir, 'fbank/train.ark')
  lab_path = join(args.data_dir, 'label/train.lab')
  write_npy(ark_path, args.data_dir, lab_path)
elif args.mode == 'test':
  ark_path = join(args.data_dir, 'fbank/test.ark')
  write_npy(ark_path, args.data_dir)
else:
  print('Illegal mode!')
