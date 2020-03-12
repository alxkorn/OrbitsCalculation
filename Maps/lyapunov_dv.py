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
    jacobis = []

    for x in [x0+dx, x0+2*dx, x0+3*dx, x0+4*dx]:
        s0 = model.get_zero_state()
        s0[0] = x
        s0[2] = z0
        dv = corrp.calc_dv(0., s0)
        s0 += dv
        states.append(s0)
        jacobis.append(model.jacobi(s0))

    i = 0
    while i < 70:
        jint = interp1d([states[j][0] for j in range(-4, 0)],
                        [jacobis[j] for j in range(-4, 0)],
                        kind='cubic',
                        fill_value='extrapolate')
        vint = interp1d([states[j][0] for j in range(-4, 0)],
                        [states[j][4] for j in range(-4, 0)],
                        kind='cubic',
                        fill_value='extrapolate')
        x = states[-1][0] + dx
        jc = jint(x)
        vy = vint(x)
        try:
            evL = op.eventSPL(model, jc, accurate=False)
            evR = op.eventSPL(model, jc, left=False, accurate=False)
            last_working_jc = jc
        except Exception as e:
           evL = op.eventSPL(model, last_working_jc, accurate=False)
           evR = op.eventSPL(model, last_working_jc, left=False, accurate=False)

        corr_unst = op.border_correction(
            model, op.unstable_direction_stm(stmmodel), evL, evR, maxt=1000.)
        corr = op.border_correction(
            model, op.y_direction(), evL, evR, maxt=1000.)
        sk = op.simple_station_keeping(model, corr, corr_unst, rev=np.pi/2)
        s0 = model.get_zero_state()
        s0[0] = x
        s0[2] = z0
        s0[4] = vy
        dv = corr.calc_dv(0., s0)

        s0 += dv
        states.append(s0)
        jacobis.append(model.jacobi(s0))
        print('%03d' % i, '%10.2f' % ((x - model.L1)*model.R), dv[4], jc)
        i += 1
    print('Done')
    states = list(map(lambda s: {'x': s[0], 'z': s[2], 'vy': s[4]}, states))
    return pd.DataFrame(states)

def do_calc(job, folder):
    model = op.crtbp3_model()
    x0 = job['x']
    z0 = job['z']
    dx = -10000/model.R

    states = run_calculations(x0, z0, dx)

    return states

def do_save(item, folder):
    job = item['job']
    filename = 'z_{}'.format(job['z'])

    item['res'].to_pickle(os.path.join(folder, filename+'.pkl'))


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
