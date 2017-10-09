import numpy as np
from tqdm import tqdm
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='parse ark (and lab) file to tfr.')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--ark', type=str, default='./data/fbank/train.ark')
parser.add_argument('--lab', type=str, default='')
parser.add_argument('--output_dir', type=str, default='./data/train.tfr')
args = parser.parse_args()

assert (not(args.mode == 'test' and args.lab != ''))

def get_intchar_map(path):
	phone2int = {}
	phone2char = {}
	with open(path) as file:
		for line in file:
			sep = line[:-1].split('\t')
			phone2int[sep[0]] = int(sep[1])
			phone2char[sep[0]] = str(sep[2])
	return phone2int, phone2char
phone2int, phone2char = get_intchar_map('./data/48phone_char.map')
int2phone = {v: k for k, v in phone2int.items()}

def get_phone_map(path):
	phone_map = {}
	with open(path) as file:
		for line in file:
			sep = line[:-1].split('\t')
			phone_map[sep[0]] = str(sep[1])
	return phone_map
phone_map = get_phone_map('./data/48_39.map')

def get_label(path):
	ark = {}
	pre_id = ''
	with open(path) as file:
		single_id_ark =[]
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
	with open(path) as file:
		single_id_ark =[]
		id = ''
		for line in file:
			line = line.strip('\n').split(' ')
			id = '_'.join(line[0].split('_')[:2])
			if (id != pre_id and len(single_id_ark) != 0):
				ark[pre_id] =  single_id_ark
				single_id_ark = []
			pre_id = id
			single_id_ark.append(np.array(line[1:], dtype=float))
		ark[id] = single_id_ark
	return ark

def get_phone(pred):
	assert(pred.ndim == 1)
	phones = []
	for i in pred:
		phones.append(phone_map[int2phone[i]])
	return phones

def write_tfr(ark_path,output_path,label_path=None):
	sequences = read_ark(ark_path)
	label_sequences = []
	if args.mode == 'train':
		label_sequences = get_label(label_path)
	print('Finish reading arks and labels.')
	def make_example(key, sequences, label_sequences=None):
		def IntList(val):
			return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
		def FloatList(val):
			return tf.train.Feature(float_list=tf.train.FloatList(value=val))

		feat = {}
		if args.mode == 'train':
			feat[key+'_label'] = IntList(label_sequences[key])
			assert(sequences.keys() == label_sequences.keys())

		for frame_id, frame in enumerate(sequences[key]):
			feat[key+'_'+str(frame_id)] = FloatList(frame)

		example = tf.train.Example(features=tf.train.Features(feature=feat))
		return example
	with open(output_path,'w') as fp:
		writer = tf.python_io.TFRecordWriter(fp.name)
		for key in tqdm(sequences):
			ex = make_example(key, sequences, label_sequences)
			writer.write(ex.SerializeToString())
		writer.close()
		print("Wrote to {}".format(fp.name))


if args.mode == 'train':
	write_tfr(args.ark, args.output_dir, args.lab)
elif args.mode == 'test':
	write_tfr(args.ark, args.output_dir)
else:
	print('Illegal mode!')