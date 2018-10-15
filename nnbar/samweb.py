import csv
from subprocess import check_output
import json

with open('get-metadata.json', 'w') as f_json:
    with open('list-files.csv') as f_csv:
        rows = csv.reader(f_csv, delimiter='\t')
        for i, row in enumerate(rows):
            filename = row[0]
            meta_data = json.loads(check_output('samweb get-metadata --json --locations {}'.format(filename), shell=True))
            f_json.write(json.dumps(meta_data) + '\n')
            if i == 100:
                print('i = {}'.format(i))
