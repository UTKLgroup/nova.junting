from subprocess import call
import glob
import os


def cerenkov_threshold_scan():
    threshold_run_numbers = [
        (100.1, 2321),
        (80.0, 2322),
        (50.1, 2323),
        (39.9, 2327),
        (35.0, 2326),
        (29.9, 2325),
        (30.0, 2368),
        (35.0, 2369)
    ]
    threshold_run_numbers = sorted(threshold_run_numbers, key=lambda x: x[0])
    for threshold_run_number in threshold_run_numbers:
        print('threshold_run_number = {}'.format(threshold_run_number))
        run_number = threshold_run_number[1]
        cmd = 'art -c fcl/cerenkovana.cosmic.fcl -s /daqdata/lariat_r00{}_*.root -T data/cerenkovana.cosmic_threshold_run_{}.root'.format(run_number, run_number)
        # print(cmd)
        call(cmd, shell=True)


def analyze_data(**kwargs):
    # run_type = 'calibration'
    # run_type = 'cosmic'
    # run_type = 'calibration_tof'
    run_type = kwargs.get('run_type', 'cosmic')
    run_number = kwargs.get('run_number', 2411)

    filenames = glob.glob('/daqdata/lariat_r00{}_*.root'.format(run_number))
    for i, filename in enumerate(filenames):
        base_name = os.path.basename(filename)
        base_name = os.path.splitext(base_name)[0]
        subrun_number = base_name.split('_')[-1]
        cmd = 'art -c fcl/cerenkovana.{}.fcl -s {} -T data/cerenkovana.{}_run_{}_{}.root'.format(run_type, filename, run_type, run_number, subrun_number)
        print('cmd = {}'.format(cmd))
        call(cmd, shell=True)

        # if i == 0:
        # cmd = 'art -c fcl/V1742Analysis.test.fcl -s {}'.format(filename)
        # call(cmd, shell=True)
        # cmd = 'mv -f V1742Analysis.root data/V1742Analysis.run_{}_{}.root'.format(run_number, subrun_number)
        # call(cmd, shell=True)
        # break

    call('hadd -f data/cerenkovana.{}_run_{}.root data/cerenkovana.{}_run_{}_*.root'.format(run_type, run_number, run_type, run_number), shell=True)
    # call('hadd -f data/V1742Analysis.run_{}.root data/V1742Analysis.run_{}_*.root'.format(run_number, run_number), shell=True)


def parse_run_log():
    with open('tmp/run.parse.log', 'w') as f_parse:
        with open('tmp/run.log') as f_log:
            last_line = None
            for line in f_log.readlines():
                if 'Opened input file' in line:
                    f_parse.write(line)

                if line.startswith('Found a pulse.'):
                    f_parse.write(last_line)
                    f_parse.write(line)
                
                if line.startswith('Begin processing'):
                    last_line = line


# parse_run_log()
analyze_data(run_type='cosmic', run_number=2440)
# cerenkov_threshold_scan()
# analyze_data()
# analyze_data(run_type='calibration_tof', run_number=2408)
# analyze_data(run_type='calibration_tof', run_number=2409)
# analyze_data(run_type='calibration_tof', run_number=2410)
# analyze_data(run_type='calibration_tof', run_number=2411)
# analyze_data(run_type='calibration_tof', run_number=2412)
# analyze_data(run_type='cosmic', run_number=2414)
# analyze_data(run_type='cosmic', run_number=2423)
# analyze_data(run_type='cosmic', run_number=2430)
# analyze_data(run_type='cosmic', run_number=2431)
# analyze_data(run_type='cosmic', run_number=2438)
# analyze_data(run_type='cosmic', run_number=2439)
# analyze_data(run_type='cosmic', run_number=2440)
