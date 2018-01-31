from subprocess import call, check_output
from ROOT import TFile


def get_diblock_count(filename):
    diblock_count = None
    tf = TFile(filename)
    for event in tf.Get('dqana/DAQHeaderSummaryAna'):
        diblock_count = event.fNDiblocks
    return diblock_count


def run_list(file_list):
    output_filename = 'diblock.txt'
    hist_filename = 'diblock.root'
    file_count = 0
    with open(output_filename, 'w') as f_out:
        with open(file_list) as f_run:
            for row in f_run.readlines():
                row = row.strip()
                xroot = check_output('samweb2xrootd {}'.format(row), shell=True).strip()
                call('nova -n 1 -c count_diblock.fcl -s {} -T {}'.format(xroot, hist_filename), shell=True)
                diblock_count = get_diblock_count(hist_filename)
                print('diblock_count = {}'.format(diblock_count))
                f_out.write('{} {}\n'.format(diblock_count, row))

                file_count += 1
                if file_count == 3:
                    break


def run_single(filename):
    hist_filename = 'diblock.root'
    xroot = check_output('samweb2xrootd {}'.format(filename), shell=True).strip()
    call('nova -n 1 -c count_diblock.fcl -s {} -T {}'.format(xroot, hist_filename), shell=True)
    diblock_count = get_diblock_count(hist_filename)
    print('diblock_count = {}'.format(diblock_count))


# get_diblock_count('diblock.root')
# run_list('prod_artdaq_S15-03-11_fd_cosmic_epochs1-3c_v1_goodruns_snapshot20161128.txt')
run_single('fardet_r00025154_s26_t02_R17-03-01-prod3reco.h_v1_data.pid.root')
