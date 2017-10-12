def get_intchar_map(path):
  phone2int = {}
  phone2char = {}
  with open(path) as file:
    for line in file:
      sep = line[:-1].split('\t')
      phone2int[sep[0]] = int(sep[1])
      phone2char[sep[0]] = str(sep[2])
  return phone2int, phone2char

def get_phone_map(path):
  phone_map = {}
  with open(path) as file:
    for line in file:
      sep = line[:-1].split('\t')
      phone_map[sep[0]] = str(sep[1])
  return phone_map

def get_phone(pred):
  assert (pred.ndim == 1)
  phones = []
  for i in pred:
    phones.append(phone_map[int2phone[i]])
  return phones

phone2int, phone2char = get_intchar_map('./data/48phone_char.map')
int2phone = {v: k for k, v in phone2int.items()}

phone_map = get_phone_map('./data/48_39.map')