import pdb

import argparse
import sys

from pathlib import Path
from jax import random
from sklearn.decomposition import PCA


from generate_data import gen_source_data
from models import init_invertible_mlp_params, invertible_mlp_fwd
from minib_train import train


def parse():
    parser = argparse.ArgumentParser(description='')

    # data generation args
    parser.add_argument('-n', type=int, default=5,
                        help="number of latent components")
    parser.add_argument('-k', type=int, default=11,
                        help="number of latent states")
    parser.add_argument('-t', type=int, default=100000,
                        help="number of time steps")
    parser.add_argument('--mix-depth', type=int, default=1,
                        help="number of mixing layers")
    parser.add_argument('--prob-stay', type=float, default=0.99,
                        help="probability of staying in a state")
    parser.add_argument('--whiten', action='store_true', default=True,
                        help="PCA whiten data as preprocessing")

    # set seeds
    parser.add_argument('--data-seed', type=int, default=0,
                        help="seed for initializing data generation")
    parser.add_argument('--mix-seed', type=int, default=0,
                        help="seed for initializing mixing mlp")
    parser.add_argument('--est-seed', type=int, default=7,
                        help="seed for initializing function estimator mlp")
    parser.add_argument('--distrib-seed', type=int, default=7,
                        help="seed for estimating distribution paramaters")
    # training & optimization parameters
    parser.add_argument('--hidden-units', type=int, default=10,
                        help="num. of hidden units in function estimator MLP")
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help="learning rate for training")
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument('--subseq-len', type=int, default=100,
                        help="length of subsequences")
    parser.add_argument('--minibatch-size', type=int, default=64,
                        help="number of subsequences in a minibatch")
    parser.add_argument('--decay-rate', type=float, default=1.,
                        help="decay rate for training (default to no decay)")
    parser.add_argument('--decay-interval', type=int, default=15000,
                        help="interval (in iterations) for full decay of LR")
    # CUDA settings
    parser.add_argument('--cuda', action='store_true', default=True,
                        help="use GPU training")
    args = parser.parse_args()
    return args


def main():
    args = parse()

    # check theoretical assumption satisfied
    assert args.k > 2*args.n, "K not set high enough for given N"

    # generate source data
    s_data, state_seq, mu, D, A = gen_source_data(args.n, args.k, args.t,
                                                  args.prob_stay,
                                                  random_seed=args.data_seed)

    # mix the sources to create observable signals
    mix_key = random.PRNGKey(args.mix_seed)
    mix_params = init_invertible_mlp_params(mix_key, args.n,
                                            args.mix_depth)
    x_data = invertible_mlp_fwd(mix_params, s_data)

    # preprocessing
    if args.whiten:
        pca = PCA(whiten=True)
        x_data = pca.fit_transform(x_data)

    # create variable dicts for training
    data_dict = {'x_data': x_data,
                 's_data': s_data,
                 'state_seq': state_seq}

    train_dict = {'mix_depth': args.mix_depth,
                  'hidden_size': args.hidden_units,
                  'learning_rate': args.learning_rate,
                  'num_epochs': args.num_epochs,
                  'subseq_len': args.subseq_len,
                  'minib_size': args.minibatch_size,
                  'decay_rate': args.decay_rate,
                  'decay_steps': args.decay_interval}

    seed_dict = {'est_mlp_seed': args.est_seed,
                 'est_distrib_seed': args.distrib_seed}

    # train HM-nICA model
    s_est, sort_idx, train_trackers, est_params = train(
        data_dict, train_dict, seed_dict,
    )

    ## save
    #Path("output/").mkdir(parents=True, exist_ok=True)
    #fileid = [str(i)+str(getattr(args, i)) for i in vars(args)]
    #fileid = '-'.join(fileid[:-5])
    #out_dict = dict()
    #out_dict['data_params'] = data_gen_dict
    #out_dict['seed_params'] = seed_dict
    #out_dict['opt_params'] = opt_dict
    #out_dict['s_est'] = s_est
    #out_dict['sort_idx'] = sort_idx
    #out_dict['train_trackers'] = train_trackers
    #out_dict['true_params'] = true_params
    #out_dict['est_params'] = est_params
    #onp.savez_compressed('output/'+'minib-'+fileid, **out_dict)


if __name__ == '__main__':
    sys.exit(main())
