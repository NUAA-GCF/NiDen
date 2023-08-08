import argparse
from logging import getLogger
import random
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch
from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import init_logger, init_seed, set_color, InputType, get_gpu_usage, early_stopping, dict2str
from recbole.trainer import Trainer
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_uniform_initialization
from time import time
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
import os

class Trainer(Trainer):

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step': epoch_idx},
                                         head='train')

            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                      + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
                valid_step += 1
                self.get_similarity_distribution(epoch_idx)
                self.get_weight_distribution(epoch_idx)
                self.get_loss_distribution(epoch_idx)
        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
        return total_loss

    def get_similarity_distribution(self,epoch_idx):
        normal_list, noise_list=self.model.similarity_distribution()
        with open('./normal_similarity.txt','a+') as f:
            f.write(f'\n{epoch_idx}:\t')
            for i in normal_list:
                f.write(f'{i}\t')
        with open('./noise_similarity.txt','a+') as f:
            f.write(f'\n{epoch_idx}:\t')
            for i in noise_list:
                f.write(f'{i}\t')

    def get_weight_distribution(self,epoch_idx):
        normal_list, noise_list=self.model.weight_distribution()
        with open('./normal_weight.txt','a+') as f:
            f.write(f'\n{epoch_idx}:\t')
            for i in normal_list:
                f.write(f'{i}\t')
        with open('./noise_weight.txt','a+') as f:
            f.write(f'\n{epoch_idx}:\t')
            for i in noise_list:
                f.write(f'{i}\t')

    def get_loss_distribution(self,epoch_idx):
        normal_list, noise_list=self.model.loss_distribution()
        with open('./normal_loss.txt','a+') as f:
            f.write(f'\n{epoch_idx}:\t')
            for i in normal_list:
                f.write(f'{i}\t')
        with open('./noise_loss.txt','a+') as f:
            f.write(f'\n{epoch_idx}:\t')
            for i in noise_list:
                f.write(f'{i}\t')

class LightSD(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(LightSD, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        self.inter_num = dataset.inter_feat.length

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.require_pow = config['require_pow']
        self.noise_rate = config['noise_rate']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.restore_list_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e', 'restore_list_e']

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def similarity_distribution(self):
        if self.restore_user_e is None or self.restore_item_e is None or self.restore_list_e is None:
            self.restore_user_e, self.restore_item_e, self.restore_list_e = self.forward()

        neighbor_users1, neighbor_items1 = torch.split(self.restore_list_e[2], [self.n_users, self.n_items])
        neighbor_users2, neighbor_items2 = torch.split(self.restore_list_e[1], [self.n_users, self.n_items])
        train_user_embs1 = neighbor_users1[self._user]
        train_item_embs2 = neighbor_items2[self._item]
        train_user_embs1 = F.normalize(train_user_embs1)
        train_item_embs2 = F.normalize(train_item_embs2)
        train_user_embs2 = neighbor_users2[self._user]
        train_item_embs1 = neighbor_items1[self._item]
        train_user_embs2 = F.normalize(train_user_embs2)
        train_item_embs1 = F.normalize(train_item_embs1)
        weight1 = torch.sum(torch.mul(train_user_embs1, train_item_embs2), dim=1)
        weight2 = torch.sum(torch.mul(train_user_embs2, train_item_embs1), dim=1)
        weight = torch.add(weight1,weight2)
        weight = (weight/2 + 1) / 2
        weight = weight.cpu().detach().numpy().tolist()
        noise_num = int(self.inter_num * self.noise_rate)
        noise_weight = weight[-noise_num:]
        noise_list = random.sample(noise_weight,1000)
        normal_num = self.inter_num - noise_num
        normal_weight = weight[:normal_num]
        normal_list = random.sample(normal_weight, 1000)
        return normal_list, noise_list

    def weight_distribution(self):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, _ = self.forward()

        u_embeddings = self.restore_user_e[self._user]
        i_embeddings = self.restore_item_e[self._item]
        weight = torch.sum(torch.mul(u_embeddings, i_embeddings), dim=1)
        weight = weight.cpu().detach().numpy().tolist()
        noise_num = int(self.inter_num * self.noise_rate)
        noise_weight = weight[-noise_num:]
        noise_list = random.sample(noise_weight, 1000)
        normal_num = self.inter_num - noise_num
        normal_weight = weight[:normal_num]
        normal_list = random.sample(normal_weight, 1000)
        return normal_list, noise_list

    def loss_distribution(self):
        if self.restore_user_e is None or self.restore_item_e is None :
            self.restore_user_e, self.restore_item_e, _ = self.forward()

        u_embeddings = self.restore_user_e[self._user]
        i_embeddings = self.restore_item_e[self._item]
        neg_item = random.choices(self._item, k=self.inter_num)
        neg_embeddings = self.restore_item_e[neg_item]
        pos_scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        weight = torch.sigmoid(neg_scores - pos_scores) #近似与将loss放缩到0-1之间
        weight = weight.cpu().detach().numpy().tolist()
        noise_num = int(self.inter_num * self.noise_rate)
        noise_weight = weight[-noise_num:]
        noise_list = random.sample(noise_weight,1000)
        normal_num = self.inter_num - noise_num
        normal_weight = weight[:normal_num]
        normal_list = random.sample(normal_weight, 1000)
        return normal_list, noise_list

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e, self.restore_list_e = None, None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings, require_pow=self.require_pow)

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, _ = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, self.restore_list_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

def run_single_model(args):
    # configurations initialization
    config = Config(
        model=LightSD,
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

    # # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = LightSD(config, train_data.dataset).to(config['device'])
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
        "embedding_size": 64,
        "n_layers": 3,
        "reg_weight": 1e-4,
        "stopping_step": 10,
        "noise_rate": 0.1,
        "gpu_id": 3
    }
    run_single_model(args)

