import orbipy as op
import pandas as pd
from orbipy import mp
import numpy as np
import os
from itertools import product
import pickle
from scipy.interpolate import interp1d


def run_calculations(x0, z0, dx):
    model = op.crtbp3_model()
    stmmodel = op.crtbp3_model(stm=True)
    left = op.eventX(model.L1-1.5e6/model.R)
    right = op.eventX(model.L1+1.5e6/model.R)

    corrp = op.border_correction(model, op.y_direction(), left, right)
    states = []
    #jacobis = []

    for x in [x0+dx, x0+2*dx, x0+3*dx, x0+4*dx]:
        s0 = model.get_zero_state()
        s0[0] = x
        s0[2] = z0
        dv = corrp.calc_dv(0., s0)
        s0 += dv
        states.append(s0)
        #jacobis.append(model.jacobi(s0))

    i = 0
    orbs = []
    fails = 0
    xx = states[-1][0]
    while i < 100:
        vint = interp1d([states[j][0] for j in range(-4, 0)],
                        [states[j][4] for j in range(-4, 0)],
                        kind='cubic',
                        fill_value='extrapolate')
        x = xx + (i+1)*dx
        vy = vint(x)
        
        jc = model.jacobi(s0)
        s0 = model.get_zero_state()
        s0[0] = x
        s0[2] = z0
        s0[4] = vy
        
        try:
            evL = op.eventSPL(model, jc, accurate=False)
            evR = op.eventSPL(model, jc, left=False, accurate=False)
            first_corr = op.border_correction(model, 
                                          op.y_direction(),
                                          [evL], [evR], 
                                          dv0=0.01, 
                                          maxt=300)
            dv = first_corr.calc_dv(0.0, s0)
            s0 += dv
            jc = model.jacobi(s0)
            evL = op.eventSPL(model, jc, accurate=False)
            evR = op.eventSPL(model, jc, left=False, accurate=False)
            first_corr = op.border_correction(model, 
                                              op.y_direction(), 
                                              [evL], [evR],
                                              dv0=1e-8,
                                              maxt=300)
            corr = op.border_correction(model, 
                                        op.unstable_direction_stm(stmmodel),
                                        [evL], [evR], 
                                        maxt=300)
            
            sk = op.strict_station_keeping(model, 
                                           first_corr, 
                                           corr,
                                           verbose=True,
                                           rev=op.eventZ(),
                                           maxdv=1e-7,
                                           maxt=300)
            
            df = sk.prop(0.0, s0, N=10)
            orbs.append(df)
            states.append(df.iloc[0].values[1:])
        except Exception as e:
            print(e)
            fails+=1
            if fails < 3:
                continue
            else:
                break

        i += 1
    with open('./spl_large_orbs/{}.bin'.format(z0), 'wb') as fp:
        pickle.dump(orbs, fp)
    return orbs

def do_calc(job, folder):
    model = op.crtbp3_model()
    x0 = job['x']
    z0 = job['z']
    dx = -10000/model.R

    states = run_calculations(x0, z0, dx)

    return states

def do_save(item, folder):
    pass

if __name__ == '__main__':
    folder = 'lyapunov_dv'
    with open('amp.pkl', 'rb') as fp:
        applicability = pickle.load(fp)
    z0_groups = applicability.groupby('z0')
    z0_values = sorted(list(z0_groups.groups.keys()))
    jobs = []
    for z0 in z0_values:
        subset = applicability.iloc[z0_groups.groups[z0]]
        if subset.shape[0] > 5:
            start_point = subset.sort_values(by='x0').iloc[5]
        else:
            start_point = subset.iloc[-1]
        x0 = start_point['x0']
        jobs.append({'x': x0, 'z': z0})
    jobs = pd.DataFrame(jobs)

    m = mp('map_1m', do_calc, do_save, folder).update_todo_jobs(jobs)

    m.run(p=4)
