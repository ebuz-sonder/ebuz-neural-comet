import random
import csv

crd = csv.DictReader(open('data_flattened.csv', 'r'))
headers = crd.fieldnames
data = list(crd)

random.seed(20211116)

data_sets = [[], [], [], []]

for row in data:
    if random.random() < .25:
        data_sets[0].append(row)
    elif random.random() < .5:
        data_sets[1].append(row)
    elif random.random() < .75:
        data_sets[2].append(row)
    else:
        data_sets[3].append(row)

for i, _ in enumerate(data_sets):
    with open(f'sample_dataset_{i+1}.csv', 'w') as of:
        cwd = csv.DictWriter(of, fieldnames=headers)
        cwd.writeheader()
        for j in range(i+1):
            cwd.writerows(data_sets[j])
