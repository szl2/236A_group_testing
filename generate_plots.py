import numpy as np
import matplotlib.pyplot as plt

import MyDecoder as decoder

def run_lp_trials(n, p, T_list, num_trial, func=decoder.lp, q=None, m=None, 
                  p_noisy=None, noise_type=None):
    Hamming_err_lists = []
    FN_lists = []
    FP_lists = []

    for i in range(num_trial):
        Hamming_err_list = []
        FN_list = []
        FP_list = []

        for T in T_list:
            if m is not None:
                X, ppl, y, A = decoder.generator_nonoverlapping(n, q, p, m, T)
            else:
                X, ppl, y = decoder.generator(n, p, T)
            
            if noise_type == 'z':
                y = decoder.add_noise_zchannel(y, p_noisy)
            elif noise_type == 'bsc':
                y = decoder.add_noise_bsc(y, p_noisy)

            if m is not None:
                ppl_pred  = func(X, y, A)
            else:
                ppl_pred = func(X, y)

            print("Solved Trial: ",i, " for T=",T)
            Hamming_err = sum(ppl_pred != ppl)
            Hamming_err_list.append(Hamming_err)

            positive_mask = (ppl == 1)
            FN = sum(ppl_pred[positive_mask] == 0)
            FN_list.append(FN)

            negative_mask = (ppl == 0)
            FP = sum(ppl_pred[negative_mask] == 1)
            FP_list.append(FP)

        Hamming_err_lists.append(Hamming_err_list)
        FN_lists.append(FN_list)
        FP_lists.append(FP_list)
    
    return Hamming_err_lists, FN_lists, FP_lists


def make_plot(n, p, Hamming_err_lists, FN_lists, FP_lists, T_list, num_trial, c=None, test='',
              m=None, q=None, p_noisy=None, noise_type=None, title=''):
    AVG_Hamming_err_list = np.average(Hamming_err_lists,axis=0)
    AVG_FN_list = np.average(FN_lists,axis=0)
    AVG_FP_list = np.average(FP_lists,axis=0)

    plt.subplot(1,3,1)
    if c is not None:
        plt.plot(T_list, AVG_Hamming_err_list, label = test, linewidth = 3)
    else:
        plt.plot(T_list, AVG_Hamming_err_list, color = c, label=test, linewidth=3)
    plt.xlabel("T (number of tests)")
    plt.ylabel("Hamming Error Rate")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.subplot(1,3,2)
    if c is not None:
        plt.plot(T_list, AVG_FN_list, label = test, linewidth = 3)
    else:
        plt.plot(T_list, AVG_FN_list, color = c, label=test, linewidth=3)
    plt.xlabel("T (number of tests)")
    plt.ylabel("False Negative")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.subplot(1,3,3)
    if c is not None:
        plt.plot(T_list, AVG_FP_list, label = test, linewidth = 3)
    else:
        plt.plot(T_list, AVG_FP_list, color = c, label=test, linewidth=3)
    plt.xlabel("T (number of tests)")
    plt.title(title)
    plt.ylabel("False Positive")
    plt.grid(True)
    plt.legend()

def plot_title(p, m=None, q=None, p_noisy=None, noise_type=None):
    title = f"p_{p}"
    if q is not None:
        title += f'_m_{m}_q_{q}'
    if noise_type is not None:
        title += f'_p_noisy_{p_noisy}_{noise_type}'
    title += '.png'
    return title


def legend_entry(p, m=None, q=None, p_noisy=None):
    title = f"p={p}"
    if q is not None:
        title += f',m={m},q={q}'
    if p_noisy is not None:
        title += f',p_noisy={p_noisy}'
    return title


if __name__ == '__main__':
    n = 1000
    T_list = [100, 200, 300, 400, 500, 600, 700]    
    num_trial = 5

    # lp
    ps = [0.1, 0.2]
    fig = plt.figure()
    for p in ps:
        out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp, q=None, m=None, 
                  p_noisy=None, noise_type=None)
        hamming_err_lists, fn_lists, fp_lists = out
        label = legend_entry(p)
        make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
                  q=None, p_noisy=None, noise_type=None, test=label)
    filename = 'lp.png'
    fig.set_size_inches(18, 4)
    plt.savefig(filename)
    plt.close()

    fig = plt.figure()
    p_noisys = [0.1, 0.2]
    # lp_noisy_z
    for p in ps:
        for p_noisy in p_noisys:
            out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp_noisy_z, q=None, m=None, 
                      p_noisy=p_noisy, noise_type='z')
            hamming_err_lists, fn_lists, fp_lists = out
            label = legend_entry(p, p_noisy=p_noisy)
            make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
                      q=None, p_noisy=p_noisy, noise_type='z', test=label)
    filename = 'lp_noisy_z.png'
    fig.set_size_inches(18, 4)
    plt.savefig(filename)
    plt.close()
            
    fig = plt.figure()
    # lp_noisy_bsc
    for p in ps:
        for p_noisy in p_noisys:
            out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp_noisy_bsc, q=None, m=None, 
                      p_noisy=p_noisy, noise_type='bsc')
            hamming_err_lists, fn_lists, fp_lists = out
            label = legend_entry(p, p_noisy=p_noisy)
            make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
                      q=None, p_noisy=p_noisy, noise_type='bsc', test=label)
    filename = 'lp_noisy_bsc.png'
    fig.set_size_inches(18, 4)
    plt.savefig(filename)
    plt.close()

    T_list = [100, 300, 500, 1000]
    ps = [0.8, 0.6]
    qs = [0.1, 0.2]
    ms = [200, 20]
    fig = plt.figure()


    # Baseline
    out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp,q=None, m=None,
                  p_noisy=None, noise_type=None)
    hamming_err_lists, fn_lists, fp_lists = out
    label = legend_entry(p, m=None, q=None, p_noisy=None)
    make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
              q=None, m=None, p_noisy=None, noise_type=None, test=label, c='black')
    # lp
    for p, q in zip(ps, qs):
        for m in ms:
                out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp_nonoverlapping, 
                          q=q, m=m, p_noisy=None, noise_type=None)
                hamming_err_lists, fn_lists, fp_lists = out
                label = legend_entry(p, m=m, q=q)
                make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
                          q=q, m=m, p_noisy=None, noise_type=None, test=label)
    filename = 'lp_nonoverlapping.png'
    fig.set_size_inches(18, 4)
    plt.savefig(filename)
    plt.close()

    p_noisys = [0.1, 0.2]
    fig = plt.figure()

    # Variations
    # lp_noisy_z
    for p, q in zip(ps, qs):
        for m in ms:
            for p_noisy in p_noisys:
                out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp_noisy_z_nonoverlapping,
                          q=q, m=m, p_noisy=p_noisy, noise_type='z')
                hamming_err_lists, fn_lists, fp_lists = out
                label = legend_entry(p, m=m, q=q, p_noisy=p_noisy)
                make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
                          q=q, m=m, p_noisy=p_noisy, noise_type='z', test=label)
    filename = 'lp_noisy_z_nonoverlapping.png'
    fig.set_size_inches(18, 4)
    plt.savefig(filename)
    plt.close()
            
    # lp_noisy_bsc
    fig = plt.figure()

    # Variations
    for p, q in zip(ps, qs):
        for m in ms:
            for p_noisy in p_noisys:
                out = run_lp_trials(n, p, T_list, num_trial, func=decoder.lp_noisy_bsc_nonoverlapping,
                          q=q, m=m, p_noisy=p_noisy, noise_type='bsc')
                hamming_err_lists, fn_lists, fp_lists = out
                label = legend_entry(p, m=m, q=q, p_noisy=p_noisy)
                make_plot(n, p, hamming_err_lists, fn_lists, fp_lists, T_list, num_trial,
                          q=q, m=m, p_noisy=p_noisy, noise_type='bsc',test=label)
    filename = 'lp_noisy_bsc_nonoverlapping.png'
    fig.set_size_inches(18, 4)
    plt.savefig(filename)
    plt.close()
