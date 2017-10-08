import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tensorflow.contrib import keras

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
	label = []
	with open(path) as file:
		for line in file:
			sep = line[:-1].split(',')
			label.append(phone2int[sep[1]])
	label = np.array(label, dtype=int)
	return label

def get_ark(path):
	ark = []
	with open(path) as file:
		for line in file:
			line = line.strip('\n').split(' ')
			ark.append(np.array(line[1:], dtype=float))
	ark = np.asarray(ark, dtype=float)
	return ark

def get_phone(pred):
	assert(pred.ndim == 1)
	phones = []
	for i in pred:
		phones.append(phone_map[int2phone[i]])
	return phones

def main():
	train = True

	if train:
		label = get_label('./data/train.lab')
		print(label.shape)
		print(label[:10])
		train_mfcc = get_ark('./data/mfcc/train.ark')
		print(train_mfcc.shape)
		print(train_mfcc[:10])
		input()
		regr = RandomForestClassifier(n_estimators=1, oob_score=True)


		cat_label = keras.utils.to_categorical(label, 48)
		regr.fit(train_mfcc, cat_label)
		print(regr.oob_score_)

		test_mfcc = get_ark('./data/mfcc/test.ark')
		prediction = regr.predict(test_mfcc)

		with open('random_forest_pred.npz','wb+') as file:
			np.save(file, prediction)
	else:
		file = open('random_forest_pred.npz','rb')
		pred = np.load(file)
		file.close()
		print(pred.shape)
		pred = pred.argmax(axis=1)
		phone_pred = get_phone(pred)

		output = open('pred.csv','w+')
		output.write('id,phone_sequence')
		test_file = open('./data/mfcc/test.ark','r')
		pre_idx = ''
		pre_phone = ''
		id = 0

		for line in test_file:
			idx = line.split(' ')[0].split('_')[:2]
			if idx != pre_idx:
				output.write('\n'+'_'.join(idx)+',')

			cur_phone = str(phone_pred[id])
			if pre_phone != cur_phone :
				output.write(cur_phone)
			pre_phone = phone_pred[id]
			pre_idx = idx
			id += 1


main()