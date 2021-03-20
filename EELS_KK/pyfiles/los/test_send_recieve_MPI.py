#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:59:11 2021

@author: isabel
"""
#hello_p2p.py
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



if rank == 0:
    save_array = np.zeros((2, size))
    save_array[:, rank] = np.array([rank, size])
    #np.save("/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/los/MPI_0", save_array)
    for i in range(1, size):
        recv_array = comm.recv(source=i)
        save_array[:,i] = recv_array
    # np.save("/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/los/MPI_0", save_array)
    print(save_array)
else:
    send_array = np.array([rank, size])
    # np.save("/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/los/MPI_" + str(rank), send_array)
    comm.send(send_array, dest=0)
    # np.save("/data/theorie/ipostmes/cluster_programs/EELS_KK/pyfiles/los/MPI_after_send_" + str(rank), send_array)
