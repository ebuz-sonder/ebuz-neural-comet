import csv
from glob import glob
import re

file_re = re.compile(r'.*/(neg|pos)/cv\d+_(\d+).txt')

data = []
for f in glob('txt_sentoken/*/*.txt'):
    f_fields = file_re.match(f)
    if f_fields:
        outcome, ID = f_fields.groups()
        text_data = open(f, 'r').read()
        data.append(dict(
            id=ID,
            outcome=outcome,
            text_data=text_data.strip()
        ))

with open('data_flattened.csv', 'w') as of:
    csv_out = csv.DictWriter(of, fieldnames=['id', 'outcome', 'text_data'])
    csv_out.writeheader()
    csv_out.writerows(data)