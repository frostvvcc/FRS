# visualize_results.py
# 运行方式：python visualize_results.py
# 会自动读取 sh_result/ml-1m.txt 和 log/ 目录，生成所有图表

import re, os, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ─────────────────────────────────────────────
# 配置区（只需改这里）
# ─────────────────────────────────────────────
RESULT_FILE = "sh_result/ml-1m.txt"
LOG_DIR     = "log"
OUT_DIR     = "figures"          # 图片保存目录
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = ['#2196F3','#4CAF50','#FF9800','#E91E63',
          '#9C27B0','#00BCD4','#8BC34A','#FF5722','#607D8B','#FFC107']

# ─────────────────────────────────────────────
# 1. 解析最终结果文件
# ─────────────────────────────────────────────
def parse_result_file(filepath):
    results = {}
    if not os.path.exists(filepath):
        print(f"[警告] 结果文件不存在：{filepath}")
        print("请先运行 train.py 跑至少一组实验！")
        return results

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            alias_m   = re.search(r'\[([^\]]+)\]', line)
            hr_m      = re.search(r'\bhr:\s*([\d.]+)', line)
            ndcg_m    = re.search(r'\bndcg:\s*([\d.]+)', line)
            lr_m      = re.search(r'\blr:\s*([\d.]+)', line)
            nb_m      = re.search(r'neighborhood_size:\s*(\d+)', line)
            rnd_m     = re.search(r'num_round:\s*(\d+)', line)
            best_m    = re.search(r'best_round:\s*(\d+)', line)
            alpha_m   = re.search(r'alpha:\s*([\d.]+)', line)

            if not (hr_m and ndcg_m):
                continue

            alias = alias_m.group(1) if alias_m else f"exp_{len(results)+1}"
            results[alias] = {
                'hr':               float(hr_m.group(1)),
                'ndcg':             float(ndcg_m.group(1)),
                'lr':               float(lr_m.group(1))  if lr_m   else None,
                'neighborhood_size':int(nb_m.group(1))    if nb_m   else None,
                'num_round':        int(rnd_m.group(1))   if rnd_m  else None,
                'best_round':       int(best_m.group(1))  if best_m else None,
                'alpha':            float(alpha_m.group(1)) if alpha_m else None,
            }
    print(f"[信息] 共读取到 {len(results)} 组实验结果")
    return results

# ─────────────────────────────────────────────
# 2. 解析日志文件（每轮 HR/NDCG 曲线）
# ─────────────────────────────────────────────
def parse_all_logs(log_dir):
    """
    读取 log/ 目录下所有日志文件
    返回：{ alias_or_filename: {'test_hr':[], 'test_ndcg':[], 'val_hr':[], 'val_ndcg':[]} }
    """
    all_curves = {}
    if not os.path.exists(log_dir):
        return all_curves

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(log_dir, fname)
        test_hr, test_ndcg, val_hr, val_ndcg = [], [], [], []
        alias_in_log = None

        with open(path, 'r') as f:
            for line in f:
                # 尝试从日志内容里找 alias
                am = re.search(r'alias[:\s]+(\S+)', line)
                if am and alias_in_log is None:
                    alias_in_log = am.group(1).strip("'\"")

                tm = re.search(r'\[Testing Round \d+\] HR = ([\d.]+), NDCG = ([\d.]+)', line)
                if tm:
                    test_hr.append(float(tm.group(1)))
                    test_ndcg.append(float(tm.group(2)))

                vm = re.search(r'\[Validating Round \d+\] HR = ([\d.]+), NDCG = ([\d.]+)', line)
                if vm:
                    val_hr.append(float(vm.group(1)))
                    val_ndcg.append(float(vm.group(2)))

        if test_hr:
            key = alias_in_log if alias_in_log else fname.replace('.txt','')
            all_curves[key] = {
                'test_hr': test_hr, 'test_ndcg': test_ndcg,
                'val_hr':  val_hr,  'val_ndcg':  val_ndcg,
            }
    print(f"[信息] 共读取到 {len(all_curves)} 个日志文件的训练曲线")
    return all_curves

# ─────────────────────────────────────────────
# 3. 图1：所有实验最终指标对比（横向柱状图）
# ─────────────────────────────────────────────
def plot_bar_comparison(results):
    if not results:
        return
    aliases    = list(results.keys())
    hr_vals    = [results[a]['hr']   for a in aliases]
    ndcg_vals  = [results[a]['ndcg'] for a in aliases]
    colors     = [COLORS[i % len(COLORS)] for i in range(len(aliases))]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(aliases)*1.5), 6))
    fig.suptitle('Final Performance Comparison (HR@10 & NDCG@10)', fontsize=13, fontweight='bold')

    for ax, vals, title, ylabel in zip(
            axes,
            [hr_vals, ndcg_vals],
            ['Hit Ratio @ 10', 'NDCG @ 10'],
            ['HR@10', 'NDCG@10']):
        bars = ax.barh(aliases, vals, color=colors, edgecolor='white', height=0.6)
        ax.set_xlabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.grid(axis='x', alpha=0.3)
        # 数值标签
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{v:.4f}', va='center', fontsize=8)
        # 最佳实验高亮
        best_idx = vals.index(max(vals))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(2.5)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, '1_bar_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[已保存] {path}")
    plt.close()

# ─────────────────────────────────────────────
# 4. 图2：训练曲线（每轮 HR 变化）
# ─────────────────────────────────────────────
def plot_training_curves(all_curves):
    if not all_curves:
        print("[跳过] 没有日志数据，无法画训练曲线")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves: HR@10 & NDCG@10 per Round', fontsize=13)

    for i, (alias, data) in enumerate(all_curves.items()):
        color = COLORS[i % len(COLORS)]
        rounds = range(1, len(data['test_hr']) + 1)
        axes[0].plot(rounds, data['test_hr'],   color=color, linewidth=1.8, label=alias)
        axes[1].plot(rounds, data['test_ndcg'], color=color, linewidth=1.8, label=alias)

    for ax, ylabel, title in zip(
            axes,
            ['HR@10', 'NDCG@10'],
            ['Hit Ratio @ 10 per Round', 'NDCG @ 10 per Round']):
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, '2_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[已保存] {path}")
    plt.close()

# ─────────────────────────────────────────────
# 5. 图3：消融实验对比（A1~A4）
# ─────────────────────────────────────────────
def plot_ablation(results):
    ablation_keys = [k for k in results if k.startswith('A')]
    if len(ablation_keys) < 2:
        print("[跳过] 消融实验组数不足（需要以 A 开头的实验 >=2 组）")
        return

    ablation_keys = sorted(ablation_keys)
    hr_vals   = [results[k]['hr']   for k in ablation_keys]
    ndcg_vals = [results[k]['ndcg'] for k in ablation_keys]
    x = range(len(ablation_keys))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()

    b1 = ax.bar( [i - 0.2 for i in x], hr_vals,   0.35,
                 color='#2196F3', alpha=0.85, label='HR@10')
    b2 = ax2.bar([i + 0.2 for i in x], ndcg_vals, 0.35,
                 color='#FF9800', alpha=0.85, label='NDCG@10')

    ax.set_xticks(list(x))
    ax.set_xticklabels(ablation_keys, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('HR@10',   color='#2196F3', fontsize=11)
    ax2.set_ylabel('NDCG@10', color='#FF9800', fontsize=11)
    ax.set_title('Ablation Study', fontsize=13, fontweight='bold')

    for bar, v in zip(b1, hr_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{v:.4f}', ha='center', fontsize=8, color='#2196F3')
    for bar, v in zip(b2, ndcg_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                 f'{v:.4f}', ha='center', fontsize=8, color='#FF9800')

    handles = [mpatches.Patch(color='#2196F3', label='HR@10'),
               mpatches.Patch(color='#FF9800', label='NDCG@10')]
    ax.legend(handles=handles, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, '3_ablation_study.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[已保存] {path}")
    plt.close()

# ─────────────────────────────────────────────
# 6. 图4：超参数敏感性——邻居数（B类）
# ─────────────────────────────────────────────
def plot_neighbor_sensitivity(results):
    b_keys = [k for k in results if k.startswith('B') and results[k]['neighborhood_size'] is not None]
    # 也把 A1（默认邻居数=10）加进来对比
    if 'A1_full_model' in results and results['A1_full_model']['neighborhood_size'] is not None:
        b_keys.append('A1_full_model')

    if len(b_keys) < 2:
        print("[跳过] 邻居数实验组数不足")
        return

    b_keys = sorted(b_keys, key=lambda k: results[k]['neighborhood_size'])
    nb_vals   = [results[k]['neighborhood_size'] for k in b_keys]
    hr_vals   = [results[k]['hr']   for k in b_keys]
    ndcg_vals = [results[k]['ndcg'] for k in b_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ax.plot(nb_vals,  hr_vals,   'b-o', linewidth=2, markersize=8, label='HR@10')
    ax2.plot(nb_vals, ndcg_vals, 'r-s', linewidth=2, markersize=8, label='NDCG@10')

    ax.set_xlabel('Neighborhood Size', fontsize=12)
    ax.set_ylabel('HR@10',   color='blue',  fontsize=11)
    ax2.set_ylabel('NDCG@10', color='red',   fontsize=11)
    ax.set_title('Sensitivity Analysis: Neighborhood Size', fontsize=13)
    ax.set_xticks(nb_vals)
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, '4_neighbor_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[已保存] {path}")
    plt.close()

# ─────────────────────────────────────────────
# 7. 图5：学习率敏感性（C类）
# ─────────────────────────────────────────────
def plot_lr_sensitivity(results):
    c_keys = [k for k in results if k.startswith('C') and results[k]['lr'] is not None]
    if 'A1_full_model' in results and results['A1_full_model']['lr'] is not None:
        c_keys.append('A1_full_model')

    if len(c_keys) < 2:
        print("[跳过] 学习率实验组数不足")
        return

    c_keys = sorted(c_keys, key=lambda k: results[k]['lr'])
    lr_vals   = [results[k]['lr']   for k in c_keys]
    hr_vals   = [results[k]['hr']   for k in c_keys]
    ndcg_vals = [results[k]['ndcg'] for k in c_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    ax.plot(lr_vals,  hr_vals,   'b-o', linewidth=2, markersize=8, label='HR@10')
    ax2.plot(lr_vals, ndcg_vals, 'r-s', linewidth=2, markersize=8, label='NDCG@10')

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('HR@10',   color='blue', fontsize=11)
    ax2.set_ylabel('NDCG@10', color='red',  fontsize=11)
    ax.set_title('Sensitivity Analysis: Learning Rate', fontsize=13)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='lower right')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, '5_lr_sensitivity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[已保存] {path}")
    plt.close()

# ─────────────────────────────────────────────
# 8. 打印汇总表
# ─────────────────────────────────────────────
def print_summary(results):
    if not results:
        return
    print("\n" + "="*72)
    print(f"{'实验名':<28} {'HR@10':>8} {'NDCG@10':>9} {'邻居数':>7} {'学习率':>8} {'最佳轮次':>8}")
    print("="*72)
    for alias in sorted(results, key=lambda k: results[k]['hr'], reverse=True):
        d = results[alias]
        print(f"{alias:<28} {d['hr']:>8.4f} {d['ndcg']:>9.4f} "
              f"{str(d.get('neighborhood_size','?')):>7} "
              f"{str(d.get('lr','?')):>8} "
              f"{str(d.get('best_round','?')):>8}")
    print("="*72)
    best = max(results.items(), key=lambda x: x[1]['hr'])
    print(f"\n🏆 最佳实验：[{best[0]}]  HR@10 = {best[1]['hr']:.4f}  NDCG@10 = {best[1]['ndcg']:.4f}\n")

# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n====== 开始生成可视化结果 ======\n")

    results   = parse_result_file(RESULT_FILE)
    all_curves = parse_all_logs(LOG_DIR)

    print_summary(results)

    plot_bar_comparison(results)       # 图1：总体对比柱状图
    plot_training_curves(all_curves)   # 图2：训练曲线
    plot_ablation(results)             # 图3：消融实验
    plot_neighbor_sensitivity(results) # 图4：邻居数敏感性
    plot_lr_sensitivity(results)       # 图5：学习率敏感性

    print(f"\n✅ 所有图表已保存到 {OUT_DIR}/ 目录")
    print("文件列表：")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  - {f}")