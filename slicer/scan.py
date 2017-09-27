from subprocess import call


def get_defname(data_sample):
    if data_sample == 'fd_cry':
        return 'prod_artdaq_R17-03-09-prod3genie.d_fd_cry_period3_v1'
    elif data_sample == 'fd_genie_nonswap':
        return 'prod_artdaq_R17-03-09-prod3genie.c_fd_genie_nonswap_fhc_nova_v08_period3_v1'
    else:
        raise Exception('Data sample {} does not exist.'.format(data_sample))

# data_sample = 'fd_cry'
data_sample = 'fd_genie_nonswap'
defname = get_defname(data_sample)
njobs = 50
files_per_job = 1
nevts = 200

for tolerance in [10]:
    for minprimdist in [4, 5, 6, 7, 8, 9, 10]:
        jobname = '{}_minprimdist_{}_tolerance_{}'.format(data_sample, minprimdist, tolerance)
        fcl_filename = '{}.fcl'.format(jobname)
        job_config_filename = '{}.config'.format(jobname)
        hist_tier = jobname
    
        with open('../job/{}'.format(fcl_filename), 'w') as f_fcl:
            with open('../RecoValidation/sliceranajob.template.fcl') as f_template:
                for row in f_template:
                    f_fcl.write(row)
            f_fcl.write('physics.producers.slicer2d.fd.MinPrimDist: {}\n'.format(minprimdist))
            f_fcl.write('physics.producers.slicermergeviews.fd.Tolerance: {}'.format(tolerance))

        with open(job_config_filename, 'w') as f_job:
            f_job.write('--jobname {}\n'.format(jobname))
            f_job.write('--defname {}\n'.format(defname))
            f_job.write('--njobs {}\n'.format(njobs))
            f_job.write('--files_per_job {}\n'.format(files_per_job))
            f_job.write('--nevts {}\n'.format(nevts))
            f_job.write('--opportunistic\n')
            f_job.write('--expected_lifetime short\n')
            f_job.write('-c {}\n'.format(fcl_filename))
            f_job.write('--testrel /nova/app/users/junting/slicer\n')
            f_job.write('--tag development\n')
            f_job.write('--dest /nova/ana/users/junting/slice\n')
            f_job.write('--copyOut\n')
            f_job.write('--histTier {}\n'.format(hist_tier))
            f_job.write('-G nova\n')

        call('../submit_nova_art.py -f {}'.format(job_config_filename), shell=True)
