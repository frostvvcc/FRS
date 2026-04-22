#!/bin/bash
# run_all_experiments.sh
# Mac 终端运行方式：bash run_all_experiments.sh

mkdir -p sh_result log

echo "========================================="
echo "A1: 完整模型（双图 + 注意力）"
echo "========================================="
python train.py --alias "A1_full_model" --num_round 50 \
  --neighborhood_size 10 --alpha 0.5 --use_attention True \
  --lr 0.1 --layers "96, 32, 16, 8"

echo "========================================="
echo "A2: 只用 item 图（alpha=1.0）"
echo "========================================="
python train.py --alias "A2_item_graph_only" --num_round 50 \
  --neighborhood_size 10 --alpha 1.0 --use_attention True \
  --lr 0.1 --layers "96, 32, 16, 8"

echo "========================================="
echo "A3: 只用 interest 图（alpha=0.0）"
echo "========================================="
python train.py --alias "A3_interest_graph_only" --num_round 50 \
  --neighborhood_size 10 --alpha 0.0 --use_attention True \
  --lr 0.1 --layers "96, 32, 16, 8"

echo "========================================="
echo "A4: 去掉注意力机制"
echo "========================================="
python train.py --alias "A4_no_attention" --num_round 50 \
  --neighborhood_size 10 --alpha 0.5 --use_attention False \
  --lr 0.1 --layers "64, 32, 16, 8"

echo "========================================="
echo "B1: 邻居数 = 5"
echo "========================================="
python train.py --alias "B1_neighbor5" --num_round 50 \
  --neighborhood_size 5 --alpha 0.5 --use_attention True \
  --lr 0.1 --layers "96, 32, 16, 8"

echo "========================================="
echo "B2: 邻居数 = 10（默认，同A1）"
echo "========================================="
echo "B2 与 A1 相同，跳过重复运行"

echo "========================================="
echo "B3: 邻居数 = 20"
echo "========================================="
python train.py --alias "B3_neighbor20" --num_round 50 \
  --neighborhood_size 20 --alpha 0.5 --use_attention True \
  --lr 0.1 --layers "96, 32, 16, 8"

echo "========================================="
echo "B4: 邻居数 = 0（阈值方式）"
echo "========================================="
python train.py --alias "B4_neighbor_threshold" --num_round 50 \
  --neighborhood_size 0 --neighborhood_threshold 1.0 \
  --alpha 0.5 --use_attention True \
  --lr 0.1 --layers "96, 32, 16, 8"

echo "========================================="
echo "C1: 学习率 = 0.05"
echo "========================================="
python train.py --alias "C1_lr005" --num_round 50 \
  --neighborhood_size 10 --alpha 0.5 --use_attention True \
  --lr 0.05 --layers "96, 32, 16, 8"

echo "========================================="
echo "C3: 学习率 = 0.2"
echo "========================================="
python train.py --alias "C3_lr02" --num_round 50 \
  --neighborhood_size 10 --alpha 0.5 --use_attention True \
  --lr 0.2 --layers "96, 32, 16, 8"

echo ""
echo "✅ 所有实验完成！运行 python visualize_results.py 查看结果"