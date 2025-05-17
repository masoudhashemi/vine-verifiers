import pickle

data = pickle.load(open("/mnt/core_llm/masoud/verifiers/verifiers/envs/imprisoned/q_table.pkl", "rb"))

data_dict = {}
for (key, value) in data.items():
    if key[0] not in data_dict:
        data_dict[key[0]] = [key[1]]
    else:
        data_dict[key[0]].append(key[1])

print(len(data_dict))

print(data_dict.keys())

print(data_dict)