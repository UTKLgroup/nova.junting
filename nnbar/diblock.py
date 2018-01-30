from subprocess import call

with open('/grid/fermiapp/products/nova/externals/novaproduction/v02.37/NULL/goodruns/fd_grl_v1_epoch1-3c.txt') as f_run:
    for row in f_run.readlines():
        print(row)
        break
