import pickle
import orbipy as op
from orbipy import mp
import numpy as np
import pandas as pd
import glob
import os
from scipy.interpolate import CubicSpline


def do_calc(job, folder):
    model = op.crtbp3_model()
    plotter = op.plotter.from_model(
        model, length_units='Gm', velocity_units='km/s')
    scale = plotter.scaler
    jobs = pickle.load(open('jobs.bin', 'rb'))
    orbit = pickle.load(open(jobs[int(job[0])], 'rb'))

    def distance(orbit, step=scale(0.5, 'd-nd'), half_rev=np.pi/2.):
        min_dist = 1e16
        max_dist = 0.0
        csx = CubicSpline(orbit['t'], orbit['x'])
        csy = CubicSpline(orbit['t'], orbit['y'])
        csz = CubicSpline(orbit['t'], orbit['z'])
        cur_time = half_rev
        last_time = orbit.iloc[-1]['t']
        while(cur_time < last_time):
            p1 = np.array(
                [csx(cur_time-half_rev), csy(cur_time-half_rev), csz(cur_time-half_rev)])
            p2 = np.array([csx(cur_time), csy(cur_time), csz(cur_time)])
            dist = np.linalg.norm(p1-p2)
            if dist > max_dist:
                max_dist = dist
            if dist < min_dist:
                min_dist = dist
            cur_time += step
        dct = dict(min_dist=min_dist, max_dist=max_dist,
                   x0=orbit.iloc[0]['x'], z0=orbit.iloc[0]['z'])
        return pd.DataFrame([dct])

    return distance(orbit, half_rev=scale(60,'d-nd'))


def do_save(item, folder):
    job = item['job']
    filename = str(int(job[0]))
    mp.mprint(filename)
    item['res'].to_pickle(os.path.join(folder, filename+'.bin'))


if __name__ == '__main__':
    folder = 'MapT60'
    jobs = pickle.load(open('jobs.bin', 'rb'))
    jobs_todo = [tuple([i]) for i, _ in enumerate(jobs)]
    m = mp('Map', do_calc, do_save, folder).update_todo_jobs(jobs_todo)
    #m.debug_run(job = jobs_todo[121])
    m.run(p=4)
