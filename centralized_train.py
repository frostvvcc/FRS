"""中心化 NCF 训练器 —— 作为联邦方案的性能上界基线。

与 train.py 不同，这里所有用户/物品在同一服务端集中训练（无联邦、无图、无 DP）。
目的：给所有联邦实验一个共同的参考系。

用法：
    python centralized_train.py --dataset 100k --num_epoch 20 \
        --metrics_json results/centralized_100k.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data import SampleGenerator
from metrics import MetronAtK


class NCF(nn.Module):
    """中心化 NCF（含序列注意力），与联邦版共享架构。"""

    def __init__(self, num_users, num_items, latent_dim=32,
                 layers=(96, 32, 16, 8), use_attention=True):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.use_attention = use_attention

        self.emb_user = nn.Embedding(num_users, latent_dim)
        self.emb_item = nn.Embedding(num_items, latent_dim)
        self.scale = latent_dim ** 0.5

        self.fc_layers = nn.ModuleList()
        for a, b in zip(layers[:-1], layers[1:]):
            self.fc_layers.append(nn.Linear(a, b))
        self.out = nn.Linear(layers[-1], 1)

    def forward(self, users, items, history):
        u = self.emb_user(users)  # (B, D)
        it = self.emb_item(items)  # (B, D)
        hist = self.emb_item(history)  # (B, L, D)

        # 注意力：query = item，keys = history
        query = it.unsqueeze(1)  # (B, 1, D)
        scores = torch.bmm(query, hist.transpose(1, 2)) / self.scale  # (B, 1, L)
        w = torch.softmax(scores, dim=-1)
        short = torch.bmm(w, hist).squeeze(1)  # (B, D)

        if self.use_attention:
            x = torch.cat([u, short, it], dim=-1)  # (B, 3D)
        else:
            x = torch.cat([u, it], dim=-1)  # (B, 2D)

        for fc in self.fc_layers:
            x = torch.relu(fc(x))
        return torch.sigmoid(self.out(x))


class InteractionDataset(Dataset):
    def __init__(self, users, items, ratings, histories):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)
        self.histories = torch.LongTensor(histories)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        return self.users[i], self.items[i], self.ratings[i], self.histories[i]


def load_ratings(dataset: str) -> tuple[pd.DataFrame, int, int]:
    path = f"data/{dataset}/ratings.dat"
    sep = '::' if dataset == 'ml-1m' else ','
    rating = pd.read_csv(path, sep=sep, header=None,
                         names=['uid', 'mid', 'rating', 'timestamp'], engine='python')

    uid_map = rating[['uid']].drop_duplicates().reset_index(drop=True)
    uid_map['userId'] = np.arange(len(uid_map))
    rating = pd.merge(rating, uid_map, on=['uid'], how='left')

    iid_map = rating[['mid']].drop_duplicates().reset_index(drop=True)
    iid_map['itemId'] = np.arange(len(iid_map))
    rating = pd.merge(rating, iid_map, on=['mid'], how='left')

    rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
    return rating, len(uid_map), len(iid_map)


def evaluate(model: NCF, eval_data, top_k=10) -> tuple[float, float]:
    test_users, test_items, neg_users, neg_items, test_hist, neg_hist = eval_data
    model.eval()
    with torch.no_grad():
        ps = model(test_users, test_items, test_hist).view(-1).tolist()
        ns = model(neg_users, neg_items, neg_hist).view(-1).tolist()

    m = MetronAtK(top_k=top_k)
    m.subjects = [
        test_users.tolist(),
        test_items.tolist(),
        ps,
        neg_users.tolist(),
        neg_items.tolist(),
        ns,
    ]
    return m.cal_hit_ratio(), m.cal_ndcg()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='100k', choices=['100k', 'ml-1m'])
    p.add_argument('--num_epoch', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--num_negative', type=int, default=12)
    p.add_argument('--history_len', type=int, default=5)
    p.add_argument('--layers', type=str, default='96,32,16,8')
    p.add_argument('--no_attention', action='store_true', default=False)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--early_stop_patience', type=int, default=3)
    p.add_argument('--metrics_json', default=None)
    p.add_argument('--result_tag', default='centralized')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = tuple(int(x) for x in args.layers.split(','))
    use_attention = not args.no_attention

    rating, n_u, n_i = load_ratings(args.dataset)
    gen = SampleGenerator(ratings=rating, history_len=args.history_len)
    model = NCF(n_u, n_i, latent_dim=args.latent_dim, layers=layers,
                use_attention=use_attention)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCELoss()

    val_data = gen.validate_data
    test_data = gen.test_data

    hr_hist, ndcg_hist, val_hr_hist, val_ndcg_hist = [], [], [], []
    best_val_hr = 0.0
    best_round = 0
    waited = 0

    for epoch in range(args.num_epoch):
        # 构造一个 epoch 的正负样本
        all_train = gen.store_all_train_data(args.num_negative)
        # 展平（中心化：所有用户拼起来）
        all_users, all_items, all_ratings, all_hists = [], [], [], []
        for u_list, i_list, r_list, h_list in zip(*all_train):
            all_users.extend(u_list)
            all_items.extend(i_list)
            all_ratings.extend(r_list)
            all_hists.extend(h_list)

        ds = InteractionDataset(all_users, all_items, all_ratings, all_hists)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

        model.train()
        losses = []
        for us, its, rs, hs in dl:
            opt.zero_grad()
            pred = model(us, its, hs).view(-1)
            loss = crit(pred, rs)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        hr, ndcg = evaluate(model, test_data)
        vhr, vndcg = evaluate(model, val_data)
        hr_hist.append(hr); ndcg_hist.append(ndcg)
        val_hr_hist.append(vhr); val_ndcg_hist.append(vndcg)
        print(f"[Epoch {epoch:02d}] train_loss={np.mean(losses):.4f} "
              f"test_HR={hr:.4f} NDCG={ndcg:.4f} val_HR={vhr:.4f}")

        if vhr > best_val_hr:
            best_val_hr = vhr
            best_round = epoch
            waited = 0
        else:
            waited += 1
            if args.early_stop_patience > 0 and waited >= args.early_stop_patience:
                print(f"Early stop at epoch {epoch}, best_epoch={best_round}")
                break

    print(f"Best test HR={hr_hist[best_round]:.4f} NDCG={ndcg_hist[best_round]:.4f} at epoch {best_round}")

    if args.metrics_json:
        os.makedirs(os.path.dirname(args.metrics_json) or '.', exist_ok=True)
        metrics = {
            'tag': args.result_tag,
            'alias': args.result_tag,
            'dataset': args.dataset,
            'method': 'centralized_ncf',
            'num_round_requested': args.num_epoch,
            'num_round_actual': len(hr_hist),
            'best_round': best_round,
            'best_test_hr': float(hr_hist[best_round]),
            'best_test_ndcg': float(ndcg_hist[best_round]),
            'best_val_hr': float(best_val_hr),
            'hr_list': [float(x) for x in hr_hist],
            'ndcg_list': [float(x) for x in ndcg_hist],
            'val_hr_list': [float(x) for x in val_hr_hist],
            'val_ndcg_list': [float(x) for x in val_ndcg_hist],
            # 中心化：没有上传成本，没有图统计
            'total_upload_bytes': 0,
            'config': vars(args),
        }
        with open(args.metrics_json, 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Metrics JSON: {args.metrics_json}")


if __name__ == '__main__':
    main()
