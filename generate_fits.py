"""
"""

import os
import joblib
from datetime import datetime

import luigi

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import curve_fit

QUANTUM_VOLUMES = {
    'ibmq_montreal': 128,
    'ibmq_guadalupe': 32,
    'ibmq_casablanca': 32,
    'ibmq_bogota': 32,
    'ibmq_rome': 32,
    'ibmq_manila': 32,
    'ibmq_manhattan': 32,
    'ibmq_toronto': 32,
    'ibmq_paris': 32,
    'ibmq_santiago': 32,
    'ibmq_athens': 32,
    'ibmq_jakarta': 16,
    'ibmq_belem': 16,
    'ibmq_quito': 16,
    'ibmq_lima': 8,
    'ibmq_16_melbourne': 8,
    'ibmqx2': 8,
}

# run parameters
MEM_FLAG = 0  # if set to 1 will select measurement error mitigated data
LOAD_FILE_NAME = 'unpacked_zne_data.joblib'
OUT_FILE_NAME = f'{"nomem_" if MEM_FLAG==0 else ""}bootstrapped_fits.joblib'
N_RESAMPLES = 50000
# FITS = ['exp,3', 'exp,7', 'lin,3', 'lin,5']
FITS = []
TARGET_BACKENDS = [
    'ibmq_lima',
    'ibmq_quito',
    'ibmq_paris',
    'ibmq_montreal',
]


def _get_backends():
    """ """
    # return [bknd for bknd in os.listdir() if 'ibmq' in bknd]
    return TARGET_BACKENDS


class MainTask(luigi.WrapperTask):

    def requires(self):
        backends = _get_backends()

        # create a directory for each backend and knot in a temporary location
        for bknd in backends:
            for knot in [
                'trefoil',
                'trefoil-twist',
                'arc-trefoil',
                'arc-trefoil-twist'
            ]:
                tmp_dir = os.path.join('analysis', 'fits', knot, bknd)
                if not os.path.isdir(tmp_dir):
                    os.makedirs(tmp_dir)

        # run for each case
        return [
            AggregateBackends(
                knot=knot,
            )
            for knot in [
                'trefoil',
                'trefoil-twist',
                'arc-trefoil',
                'arc-trefoil-twist'
            ]
        ]


class AggregateBackends(luigi.Task):
    knot = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_filepath = os.path.join(
            'analysis',
            'fits',
            self.knot,
            OUT_FILE_NAME,
        )

    def requires(self):
        return FitAllBackends(knot=self.knot)

    def output(self):
        return luigi.LocalTarget(self.out_filepath)

    def run(self):
        backends = _get_backends()

        df = None
        for bknd in backends:

            load_filepath = os.path.join(
                'analysis',
                'fits',
                self.knot,
                bknd,
                OUT_FILE_NAME,
            )
            tmp_df = joblib.load(load_filepath)

            tmp_df['backend_name'] = bknd

            if df is None:
                df = tmp_df
            else:
                df = df.append(tmp_df, ignore_index=True)

        # save to disk
        joblib.dump(df, self.out_filepath, compress=True)


class FitAllBackends(luigi.WrapperTask):
    knot = luigi.Parameter()

    def requires(self):
        backends = _get_backends()

        return [
            FitZNEData(
                knot=self.knot,
                backend=bknd,
            )
            for bknd in backends
        ]


def exp_fit(x, a, b):
    return a*np.exp(-b*x)


def lin_fit(x, a, b):
    return a + b*x


def make_fit(fit_name, fit_df):
    """ """

    fit_type, max_c = fit_name.split(',')

    if fit_type == 'lin':
        fitter = lin_fit
        bounds = (-np.inf, np.inf)
    elif fit_type == 'exp':
        fitter = exp_fit
        bounds = ([-1., 0], [1., np.inf])

    # select out scale factors up to max set by fit_name
    _df = fit_df.loc[
        (fit_df['scale_factor'] < int(max_c)+1)
    ]

    try:
        popt, pcov = curve_fit(
            fitter,
            _df['scale_factor'].values,
            _df['mean_zne'].values,
            sigma=_df['std_zne'].values,
            bounds=bounds,
            # absolute_sigma=True,
        )
    except RuntimeError:
        popt = np.array([np.nan, np.nan])
        pcov = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    return popt, pcov


def df_wide_to_long(wide_df):
    """ """

    wide_df['id'] = wide_df.index
    long_df = pd.wide_to_long(
        wide_df,
        stubnames=['mean_zne', 'std_zne'],
        sep='_',
        i='id', j='scale_factor',
    )
    long_df = long_df.reset_index()
    long_df.dropna(axis=0, subset=['mean_zne', 'std_zne'], inplace=True,)
    long_df.drop(columns='id', inplace=True)
    return long_df


class FitZNEData(luigi.Task):
    knot = luigi.Parameter()
    backend = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_filepath = os.path.join(
            'analysis',
            'fits',
            self.knot,
            self.backend,
            OUT_FILE_NAME,
        )

        self.load_filepath = os.path.join(
            'analysis',
            'fits',
            self.knot,
            self.backend,
            LOAD_FILE_NAME,
        )

    def requires(self):
        return PrepareZNEData(
            knot=self.knot,
            backend=self.backend,
        )

    def output(self):
        return luigi.LocalTarget(self.out_filepath)

    def run(self):
        """ """

        def _long_resample_func(x):
            x = x.sample(frac=1, replace=True)
            x = x.drop(columns=['part', 'scale_factor'])
            return x

        fit_df = pd.DataFrame(
            columns=['resample_style', 'part', 'target_val', 'raw_mean']+FITS,
            index=pd.RangeIndex(N_RESAMPLES*2*2),
        )
        fit_df_idx = 0

        # load data
        wide_df = joblib.load(self.load_filepath)
        wide_df = wide_df.loc[(wide_df['meas_err_mit'] == MEM_FLAG)]
        long_df = df_wide_to_long(wide_df)
        target_vals = long_df.groupby(
            ['part'])['target_val'].mean().reset_index()

        # fit to real and imag part
        for part in ['real', 'imag']:
            for _ in range(N_RESAMPLES):
                for resample_style in ['by_scale_factor', 'by_experiment']:

                    # row header fields
                    row = [
                        resample_style,
                        part,
                        target_vals.loc[
                            target_vals['part'] == part
                        ]['target_val'].values[0]
                    ]

                    if resample_style == 'by_scale_factor':
                        tmp = long_df.loc[long_df['part'] == part]
                        g = tmp.groupby(['scale_factor'])
                        resample_df = g.apply(_long_resample_func).reset_index()
                    elif resample_style == 'by_experiment':
                        tmp = wide_df.loc[wide_df['part'] == part]
                        resample_df = tmp.sample(
                            frac=1, replace=True, ignore_index=True
                        )
                        resample_df = df_wide_to_long(resample_df)
                    else:
                        raise ValueError(
                            f'resample style: {resample_style}, not recognised.'
                        )

                    # raw mean
                    row += [resample_df.loc[
                        (resample_df['scale_factor'] == 1)
                    ]['mean_zne'].mean()]

                    # make fit
                    for fname in FITS:
                        popt, _ = make_fit(fname, resample_df, )
                        row += [popt[0]]

                    fit_df.loc[fit_df_idx] = row
                    fit_df_idx += 1

        # save to disk
        joblib.dump(fit_df, self.out_filepath, compress=True)


def clean_and_process_experimental_data(
    df,
    date_int,
    backend_name,
):
    """ """

    # add in useful extra fields
    df['date_int'] = date_int
    df['date_str'] = datetime.fromtimestamp(date_int).strftime(
        "%d/%m/%y")
    df['QV'] = QUANTUM_VOLUMES[backend_name]

    # select out only the data we want -- in this case the length 7 and 9
    # series of ZNE data
    df = df.loc[
        (
            df['err_mit_strat'].apply(
                lambda x: ('zne' in x) and (('7' in x) or ('9' in x))
            )
        )
    ]

    # drop 'meas_err_mit' col
    # df = df.loc[(df['meas_err_mit'] == MEM_FLAG)]
    # df.drop(columns='meas_err_mit', inplace=True)

    # correct naming error on std cols
    for zne_stretch in [1, 3, 5, 7, 9]:
        df[f'std_zne_{zne_stretch}'] = np.sqrt(df[f'std_zne_{zne_stretch}'])

    # rename 'err_mit_strat' col
    df.rename(columns={'err_mit_strat': 'data_src'}, inplace=True)

    return df


class PrepareZNEData(luigi.Task):
    """
    Here we are dropping all of the data apart from:
        meas_err_mit == MEM_FLAG
        err_mit_strat == 'zne,linear,*'
    and we are averaging this data over different days. These averaged data
    series are then refit using either Linear, Richardson or Exponential
    extrapolation.
    """

    knot = luigi.Parameter()
    backend = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.out_filepath = os.path.join(
            'analysis',
            'fits',
            self.knot,
            self.backend,
            LOAD_FILE_NAME,
        )

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(self.out_filepath)

    def run(self):
        df = None

        dates_dirs = [d for d in os.listdir(self.backend) if d != '.DS_Store']
        for ddir in dates_dirs:

            try:
                tmp_df = joblib.load(
                    os.path.join(
                        self.backend,
                        ddir,
                        self.knot,
                        LOAD_FILE_NAME,
                    )
                )
            except FileNotFoundError:
                continue

            # clean and process dataset
            tmp_df = clean_and_process_experimental_data(
                tmp_df, int(ddir), self.backend)

            if df is None:
                df = tmp_df
            else:
                df = df.append(tmp_df, ignore_index=True)

        # save to disk
        joblib.dump(df, self.out_filepath, compress=True)
