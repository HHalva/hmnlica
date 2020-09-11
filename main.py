import argparse
import pdb
import numpy as onp
from pathlib import Path
from jax.config import config

from jax_sim_data import gen_source_data
from minib_train import train

# enable 64bit float points
config.update('jax_enable_x64', True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data generation args
    parser.add_argument('-n', type=int, default=5,
                        help="number of latent components")
    parser.add_argument('-k', type=int, default=11,
                        help="number of latent states")
    parser.add_argument('-t', type=int, default=100000,
                        help="number of time steps")
    parser.add_argument('-m', type=int, default=5,
                        help="number of mixing layers")
    parser.add_argument('-p', '--prob-stay', type=float, default=0.99,
                        help="probability of staying in a state")
    # set seeds
    parser.add_argument('--ds', '--data-seed', type=int, default=0,
                        help="seed for initializing data generation")
    parser.add_argument('--mk', '--mix-key', type=int, default=0,
                        help="jax key for initializing mixing mlp")
    parser.add_argument('--ek', '--est-key', type=int, default=1,
                        help="jax key for initializing estimator mlp")
    parser.add_argument('--ps', '--distrib-seed', type=int, default=0,
                        help="seed for estimating distribution paramaters")

    # training & optimization parameters
    parser.add_argument('-u', '--hidden-units', type=int, default=100,
                        help="number of hidden units in estimator MLP layer")
    parser.add_argument('-l', '--learning-rate', type=float, default=3e-4,
                        help="learning rate for training")
    parser.add_argument('-e', '--num-epochs', type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument('--sl', '--subseq-len', type=int, default=100,
                        help="length of subsequences")
    parser.add_argument('--ms', '--minibatch-size', type=int, default=64,
                        help="number of subsequences in a minibatch")
    parser.add_argument('-d', '--decay-rate', type=float, default=1.,
                        help="decay rate for training")
    parser.add_argument('-i', '--decay-interval', type=int, default=10000,
                        help="interval for full decay of LR")
    # misc
    parser.add_argument('-v', '--vis-pre', action='store_true', default=False,
                        help="visualize distributions pre training")
    parser.add_argument('-w', '--vis-train', action='store_true', default=False,
                        help="visualize distributions during training")
    parser.add_argument('--tm', '--true-mix', action='store_true', default=False,
                        help="force estimator mlp params to true parameters")
    parser.add_argument('--td', '--true-distrib', action='store_true', default=False,
                        help="force estimated distrib. parameters to true")
    parser.add_argument('-c', '--cuda', action='store_true', default=False,
                        help="use GPU training")
    args = parser.parse_args()

    # check theoretical assumption
    assert args.k > 2*args.n, "K not set high enough for given N"

    # set up data generation variables
    data_gen_dict = {'N': args.n,
                     'K': args.k,
                     'T': args.t,
                     'p_stay': args.prob_stay,
                     'mix_depth': args.m}

    seed_dict = {'data_seed': args.ds,
                 'key_mix_mlp': args.mk,
                 'key_est_mlp': args.ek,
                 'seed_est_distrib': args.ps}

    opt_dict = {'hidden_size': args.hidden_units,
                'learning_rate': args.learning_rate,
                'num_epochs': args.num_epochs,
                'subseq_len': args.sl,
                'minib_size': args.ms,
                'decay_rate': args.decay_rate,
                'decay_steps': args.decay_interval}

    # train
    s_est, sort_idx, train_trackers, true_params, est_params = train(
        data_gen_dict, opt_dict, seed_dict,
        viz_pre=args.vis_pre, viz_train=args.vis_train,
        use_true_mix=args.tm, use_true_distrib=args.td
    )

    # save
    Path("output/").mkdir(parents=True, exist_ok=True)
    fileid = [str(i)+str(getattr(args, i)) for i in vars(args)]
    fileid = '-'.join(fileid[:-5])
    out_dict = dict()
    out_dict['data_params'] = data_gen_dict
    out_dict['seed_params'] = seed_dict
    out_dict['opt_params'] = opt_dict
    out_dict['s_est'] = s_est
    out_dict['sort_idx'] = sort_idx
    out_dict['train_trackers'] = train_trackers
    out_dict['true_params'] = true_params
    out_dict['est_params'] = est_params
    onp.savez_compressed('output/'+'minib-'+fileid, **out_dict)
