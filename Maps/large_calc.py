import orbipy as op
from orbipy import mp
import numpy as np
import pandas as pd
import pickle
import os
from itertools import product

# Calculate orbit for 100 revolutions and return spacecraft states in DataFrame
def do_calc(job, folder):
    model = op.crtbp3_model()
    plotter = op.plotter.from_model(model, length_units='nd')
    scale = plotter.scaler
    stmmodel = op.crtbp3_model(stm=True)
    pmodel = op.crtbp3_model() # precision model
    pmodel.integrator.set_params(max_step=scale(1, 'd-nd'))
    s0 = model.get_zero_state()
    s0[0] = job['x']
    s0[2] = job['z']
    s0[4] = job['vy']
    df0 = model.prop(s0,0.0,2*np.pi)

    xmax = df0['x'].max()
    left = op.eventX(df0['x'].min()-scale(200000, 'km-nd'))
    right = op.eventX(xmax + scale(30000,'km-nd'))
    corr_unst = op.border_correction(model, op.unstable_direction_stm(stmmodel), left, right, maxt=1000.)
    corr = op.border_correction(model, op.y_direction(), left, right, maxt=1000.)
    sk = op.simple_station_keeping(model, corr, corr_unst, rev=0.5*np.pi)
    df = sk.prop(0.0, s0, 40)

    return df


def do_save(item, folder):
    job = item['job']
    filename = 'orbit_{:.10f}_{:.10f}'.format(job['x'], job['z'])
    # mp.mprint(filename)
    item['res'].to_pickle(os.path.join(folder, filename+'.pkl'))

if __name__ == '__main__':       
    folder = 'large_orbs'
    
    with open('../OrbitsCalculation/Notebooks/after_lyapunov_dv_jobs.pkl', 'rb') as fp:
        jobs_todo = pickle.load(fp)
    
    m = mp('map_1m', do_calc, do_save, folder).update_todo_jobs(jobs_todo)

    m.run(p=4)