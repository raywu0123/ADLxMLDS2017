import i2v
from PIL import Image
import operator
from tqdm import tqdm
import os

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

hair_types = [  'orange hair', 'white hair', 'aqua hair', 'grey hair',
                'green hair', 'red hair', 'purple hair', 'pink hair',
                'blue hair', 'black hair', 'brown hair', 'blonde hair']

eye_types = [   'grey eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

output_file = open('./tags2.csv', 'w+')
base_num = 33431
data_dir = '../data/faces2/'
file_num = len(os.listdir(data_dir))

for i in tqdm(range(file_num)):
    img = Image.open(data_dir+str(base_num + i)+'.jpg')

    hair_scores = illust2vec.estimate_specific_tags([img], hair_types)
    eye_scores = illust2vec.estimate_specific_tags([img], eye_types)
    hair_type = max(hair_scores[0].items(), key=operator.itemgetter(1))[0]
    if hair_type == 'grey hair':
        hair_type = 'gray hair'
    eye_type = max(eye_scores[0].items(), key=operator.itemgetter(1))[0]
    output_file.write(str(base_num + i) + ',' + hair_type + ',' + eye_type + '\n')

output_file.close()