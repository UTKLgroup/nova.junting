from rootalias import *
import csv

with open('data/F1ch200000.txt') as f_csv:
    reader = csv.reader(f_csv, delimiter='\t')
    # row_count = sum(1 for row in reader) - 1
    # print('row_count = {}'.format(row_count))
    # f_csv.seek(0)
    next(reader)
    row_count = 0
    charge_entries = []
    for row in reader:
        print('row = {}'.format(row))
        charge = float(row[0]) / 50. # Coulomb
        entry = float(row[1])
        charge_entries.append((charge, entry))

    gr=new TH1F(addr.c_str(),addr.c_str(),i-1,-x[i-2]*1E9,-x[0]*1E9);
    
