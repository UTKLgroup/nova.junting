# samweb list-files --fileinfo "online.detector fardet AND data_tier='raw' AND data_stream ddnnbar" > tmp/samweb.ddnnbar.txt
# samweb list-files --fileinfo "online.detector fardet AND data_tier='raw' AND data_stream ddnnbar AND end_time >= 2018-11-01 AND end_time <= 2018-11-02" > tmp/samweb.ddnnbar.txt
samweb list-files --fileinfo "online.detector fardet AND data_tier='raw' AND data_stream ddnnbar AND end_time >= 2018-11-01" > list-files.2019_02_20.csv
