import argparse
from logging import getLogger

import torch
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.utils import init_logger, init_seed, set_color
from recbole.trainer import Trainer
import recbole.model.general_recommender.lightgcn as lgcn

def run_single_model(args):
    # configurations initialization
    config = Config(
        model=lgcn.LightGCN,
        dataset=args.dataset,
        config_file_list=args.config_file_list,
        config_dict=args.config_dict
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    #
    # # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = lgcn.LightGCN(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=False, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp', help='The datasets can be: ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml'
    ]
    if args.dataset in ['ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba', 'ml-100k', 'last-fm', 'yelp2018', 'purchase']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)
    args.config_dict = {
        "gpu_id": 1
    }

    run_single_model(args)

