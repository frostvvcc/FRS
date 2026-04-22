import matplotlib.pyplot as plt

# 直接复制你日志里的数组
hit_ratio_list = [0.1049, 0.0986, 0.1145, 0.1124, 0.1527, 0.1866, 0.2216, 0.2523, 0.3001, 0.3170, 0.3382, 0.3531, 0.3870, 0.3753, 0.4103, 0.4337, 0.4209, 0.4156, 0.4284, 0.4369, 0.4485, 0.4326, 0.4506, 0.4602, 0.4687, 0.4591, 0.4782, 0.4814, 0.5047, 0.4984, 0.5143, 0.5026, 0.5206, 0.5238, 0.5185, 0.5249, 0.5217, 0.5312, 0.5503, 0.5323, 0.5673, 0.5546, 0.5705, 0.5524, 0.5737, 0.5524, 0.5577, 0.5482, 0.5630, 0.5662, 0.5874, 0.5790, 0.5885, 0.5970, 0.6002, 0.6002, 0.6108, 0.6108, 0.6012, 0.6065, 0.6065, 0.6182, 0.6256, 0.6139, 0.6108, 0.6161, 0.6171, 0.6171, 0.6139, 0.6108, 0.6129, 0.6118, 0.6150, 0.6341, 0.6320, 0.6320, 0.6182, 0.6352, 0.6394, 0.6246, 0.6288, 0.6267, 0.6426, 0.6447, 0.6394, 0.6436, 0.6532, 0.6564, 0.6458, 0.6415, 0.6468, 0.6532, 0.6405, 0.6479, 0.6436, 0.6447, 0.6458, 0.6426, 0.6532, 0.6479]

# 设置图片大小和分辨率
plt.figure(figsize=(10, 6), dpi=150)

# 绘制折线
rounds = range(len(hit_ratio_list))
plt.plot(rounds, hit_ratio_list, marker='', linestyle='-', color='#d62728', linewidth=2, label='HR@10')

# 标记最高点
best_round = 96  # 你日志里写的 96 轮
best_hr = hit_ratio_list[best_round]
plt.plot(best_round, best_hr, 'k*', markersize=12)
plt.annotate(f'Best HR: {best_hr:.4f}',
             xy=(best_round, best_hr),
             xytext=(best_round-20, best_hr-0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

# 设置学术图表样式
plt.title('Model Convergence on MovieLens-100k', fontsize=14)
plt.xlabel('Communication Rounds', fontsize=12)
plt.ylabel('Hit Ratio (HR@10)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', fontsize=12)

# 保存为高清图片，直接插入 Word！
plt.savefig('learning_curve.png', bbox_inches='tight')
plt.show()