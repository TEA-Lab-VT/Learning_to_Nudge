import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import argparse
import copy

# 从之前的文件中导入必要的类和函数
from models import NCBF, Normalizer  # 导入PyTorch版本的模型
from config import *  # 导入配置参数
from helpers import normalize_data  # 导入helpers.py中的归一化函数

def parse_args():
    parser = argparse.ArgumentParser(description="评估NCBF模型并找出高损失样本")
    parser.add_argument('--model', type=str, default="results_0724/model_iter_100.pt", help='训练好的模型路径')
    parser.add_argument('--data', type=str, default="ncbf_trajectories/trajectories_reformat_relative.npz", help='格式化后的数据路径')
    parser.add_argument('--h_obj_normalizer', default="results_0724/local_normalizer", help='h_obj归一化器路径')
    parser.add_argument('--ee_normalizer', default="results_0724/arm_normalizer", help='ee归一化器路径')
    parser.add_argument('--top_n', type=int, default=250, help='要分析的高损失样本数量')
    parser.add_argument('--output', type=str, default='evaluation_results', help='结果保存目录')
    return parser.parse_args()

def load_formatted_data(data_filepath):
    """
    加载格式化后的训练数据
    """
    print(f"正在从{data_filepath}加载格式化训练数据...")
    data = np.load(data_filepath, allow_pickle=True)
    
    # 加载原始数据
    safe_local_xs = data['safe_local_xs']
    safe_local_nxs = data['safe_local_nxs']
    safe_ee_xs = data['safe_ee_xs']
    safe_ee_nxs = data['safe_ee_nxs']
    unsafe_local_xs = data['unsafe_local_xs']
    unsafe_ee_xs = data['unsafe_ee_xs']
    
    print(f"加载的训练数据:")
    print(f"  safe_local_xs 形状: {safe_local_xs.shape}")
    print(f"  safe_local_nxs 形状: {safe_local_nxs.shape}")
    print(f"  safe_ee_xs 形状: {safe_ee_xs.shape}")
    print(f"  safe_ee_nxs 形状: {safe_ee_nxs.shape}")
    print(f"  unsafe_local_xs 形状: {unsafe_local_xs.shape}")
    print(f"  unsafe_ee_xs 形状: {unsafe_ee_xs.shape}")
    
    # 提取特征数据 (只取前4个元素: tilt, rel_x, rel_y, rel_z)
    print("提取模型评估所需的特征...")
    
    # 处理safe_local_xs
    safe_h_obj_features = np.zeros((safe_local_xs.shape[0], safe_local_xs.shape[1], 4), dtype=np.float32)
    for i in range(safe_local_xs.shape[0]):
        for j in range(safe_local_xs.shape[1]):
            safe_h_obj_features[i, j] = safe_local_xs[i, j, :4].astype(np.float32)
    
    # 处理safe_local_nxs
    safe_h_obj_nxs_features = np.zeros((safe_local_nxs.shape[0], safe_local_nxs.shape[1], 4), dtype=np.float32)
    for i in range(safe_local_nxs.shape[0]):
        for j in range(safe_local_nxs.shape[1]):
            safe_h_obj_nxs_features[i, j] = safe_local_nxs[i, j, :4].astype(np.float32)
            
    # 处理unsafe_local_xs
    unsafe_h_obj_features = np.zeros((unsafe_local_xs.shape[0], unsafe_local_xs.shape[1], 4), dtype=np.float32)
    for i in range(unsafe_local_xs.shape[0]):
        for j in range(unsafe_local_xs.shape[1]):
            unsafe_h_obj_features[i, j] = unsafe_local_xs[i, j, :4].astype(np.float32)
    
    # 确保末端执行器数据是浮点类型
    safe_ee_xs = safe_ee_xs.astype(np.float32)
    safe_ee_nxs = safe_ee_nxs.astype(np.float32)
    unsafe_ee_xs = unsafe_ee_xs.astype(np.float32)
    
    # 保存原始数据（包括额外信息）用于后续分析
    original_data = {
        'safe_local_xs': safe_local_xs,
        'safe_local_nxs': safe_local_nxs,
        'unsafe_local_xs': unsafe_local_xs
    }
    
    return safe_h_obj_features, safe_h_obj_nxs_features, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_features, unsafe_ee_xs, original_data

def load_normalizers(h_obj_normalizer_path, ee_normalizer_path):
    """
    加载训练好的归一化器
    """
    print(f"从{h_obj_normalizer_path}和{ee_normalizer_path}加载归一化器...")
    
    h_obj_normalizer = Normalizer(4)  # 4维特征：tilt, rel_x, rel_y, rel_z
    h_obj_normalizer.load_model(h_obj_normalizer_path)
    
    ee_normalizer = Normalizer(2)  # 2维特征：ee_x, ee_y
    ee_normalizer.load_model(ee_normalizer_path)
    
    return h_obj_normalizer, ee_normalizer

def evaluate_individual_losses(model, normalized_data, device, margin_threshold=0.01, derivative_threshold=0.05):
    """
    计算每个样本的单独损失值
    """
    safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs = normalized_data
    
    model.eval()
    model.to(device)
    
    # 创建结果存储
    safe_losses = []
    unsafe_losses = []
    
    # 评估安全样本
    print("评估安全样本...")
    for i in tqdm(range(len(safe_h_obj_xs))):
        # 准备输入数据
        safe_h_obj_x = torch.tensor(safe_h_obj_xs[i], dtype=torch.float32).unsqueeze(0).to(device)
        safe_h_obj_nx = torch.tensor(safe_h_obj_nxs[i], dtype=torch.float32).unsqueeze(0).to(device)
        safe_ee_x = torch.tensor(safe_ee_xs[i], dtype=torch.float32).unsqueeze(0).to(device)
        safe_ee_nx = torch.tensor(safe_ee_nxs[i], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            safe_val = model([safe_h_obj_x, safe_ee_x])
            safe_nval = model([safe_h_obj_nx, safe_ee_nx])
            
            # 计算各种损失
            pos_sign_loss = F.relu(safe_val + margin_threshold).item()
            derivative_loss = F.relu(safe_nval + (model.CBF_gamma - 1) * safe_val + margin_threshold).item()
            bounding_loss = F.relu((safe_nval - safe_val - derivative_threshold)).item()
            
            # 总损失
            total_loss = pos_sign_loss + derivative_loss + bounding_loss
            
            # 存储损失和CBF值
            safe_losses.append({
                'index': i,
                'total_loss': total_loss,
                'pos_sign_loss': pos_sign_loss,
                'derivative_loss': derivative_loss,
                'bounding_loss': bounding_loss,
                'cbf_val': safe_val.item(),
                'cbf_next_val': safe_nval.item()
            })
    
    # 评估不安全样本
    print("评估不安全样本...")
    for i in tqdm(range(len(unsafe_h_obj_xs))):
        # 准备输入数据
        unsafe_h_obj_x = torch.tensor(unsafe_h_obj_xs[i], dtype=torch.float32).unsqueeze(0).to(device)
        unsafe_ee_x = torch.tensor(unsafe_ee_xs[i], dtype=torch.float32).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.no_grad():
            unsafe_val = model([unsafe_h_obj_x, unsafe_ee_x])
            
            # 计算损失
            neg_sign_loss = F.relu(-unsafe_val + margin_threshold).item()
            
            # 总损失
            total_loss = neg_sign_loss
            
            # 存储损失和CBF值
            unsafe_losses.append({
                'index': i,
                'total_loss': total_loss,
                'neg_sign_loss': neg_sign_loss,
                'cbf_val': unsafe_val.item()
            })
    
    return safe_losses, unsafe_losses

def analyze_high_loss_samples(safe_losses, unsafe_losses, top_n, original_data):
    """
    分析高损失样本
    """
    # 排序安全样本损失
    sorted_safe_losses = sorted(safe_losses, key=lambda x: x['total_loss'], reverse=True)
    
    # 排序不安全样本损失
    sorted_unsafe_losses = sorted(unsafe_losses, key=lambda x: x['total_loss'], reverse=True)
    
    # 获取top_n高损失样本
    top_safe_losses = sorted_safe_losses[:top_n]
    top_unsafe_losses = sorted_unsafe_losses[:top_n]
    
    # 计算损失统计信息
    safe_total_losses = [item['total_loss'] for item in safe_losses]
    unsafe_total_losses = [item['total_loss'] for item in unsafe_losses]
    
    safe_stats = {
        'mean': np.mean(safe_total_losses),
        'median': np.median(safe_total_losses),
        'max': np.max(safe_total_losses),
        'min': np.min(safe_total_losses),
        'percentile_95': np.percentile(safe_total_losses, 95),
        'percentile_99': np.percentile(safe_total_losses, 99)
    }
    
    unsafe_stats = {
        'mean': np.mean(unsafe_total_losses),
        'median': np.median(unsafe_total_losses),
        'max': np.max(unsafe_total_losses),
        'min': np.min(unsafe_total_losses),
        'percentile_95': np.percentile(unsafe_total_losses, 95),
        'percentile_99': np.percentile(unsafe_total_losses, 99)
    }
    
    # 分析高损失安全样本中的贡献比例
    safe_contribution = {
        'pos_sign': sum(item['pos_sign_loss'] for item in top_safe_losses) / sum(item['total_loss'] for item in top_safe_losses) if sum(item['total_loss'] for item in top_safe_losses) > 0 else 0,
        'derivative': sum(item['derivative_loss'] for item in top_safe_losses) / sum(item['total_loss'] for item in top_safe_losses) if sum(item['total_loss'] for item in top_safe_losses) > 0 else 0,
        'bounding': sum(item['bounding_loss'] for item in top_safe_losses) / sum(item['total_loss'] for item in top_safe_losses) if sum(item['total_loss'] for item in top_safe_losses) > 0 else 0
    }
    
    # 提取高损失样本的特征信息
    high_loss_safe_features = []
    for item in top_safe_losses:
        idx = item['index']
        # 提取该样本的特征信息（只取当前时刻的tilt, rel_x, rel_y, rel_z）
        features = original_data['safe_local_xs'][idx][-1][:4]
        high_loss_safe_features.append({
            'index': idx,
            'loss': item['total_loss'],
            'tilt': features[0],
            'rel_x': features[1],
            'rel_y': features[2],
            'rel_z': features[3],
            'cbf_val': item['cbf_val'],
            'cbf_next_val': item['cbf_next_val']
        })
    
    high_loss_unsafe_features = []
    for item in top_unsafe_losses:
        idx = item['index']
        # 提取该样本的特征信息
        features = original_data['unsafe_local_xs'][idx][-1][:4]
        high_loss_unsafe_features.append({
            'index': idx,
            'loss': item['total_loss'],
            'tilt': features[0],
            'rel_x': features[1],
            'rel_y': features[2],
            'rel_z': features[3],
            'cbf_val': item['cbf_val']
        })
    
    return {
        'top_safe_losses': top_safe_losses,
        'top_unsafe_losses': top_unsafe_losses,
        'safe_stats': safe_stats,
        'unsafe_stats': unsafe_stats,
        'safe_contribution': safe_contribution,
        'high_loss_safe_features': high_loss_safe_features,
        'high_loss_unsafe_features': high_loss_unsafe_features
    }

def plot_loss_distribution(safe_losses, unsafe_losses, output_dir):
    """
    绘制损失分布图
    """
    safe_total_losses = [item['total_loss'] for item in safe_losses]
    unsafe_total_losses = [item['total_loss'] for item in unsafe_losses]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(safe_total_losses, bins=50, alpha=0.7, label='安全样本')
    plt.xlabel('损失值')
    plt.ylabel('频率')
    plt.title('安全样本的损失分布')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(unsafe_total_losses, bins=50, alpha=0.7, label='不安全样本')
    plt.xlabel('损失值')
    plt.ylabel('频率')
    plt.title('不安全样本的损失分布')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_distribution.png'))
    plt.close()
    
    # 绘制CBF值分布图
    safe_cbf_vals = [item['cbf_val'] for item in safe_losses]
    unsafe_cbf_vals = [item['cbf_val'] for item in unsafe_losses]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(safe_cbf_vals, bins=50, alpha=0.7, label='安全样本')
    plt.xlabel('CBF值')
    plt.ylabel('频率')
    plt.title('安全样本的CBF值分布')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(unsafe_cbf_vals, bins=50, alpha=0.7, label='不安全样本')
    plt.xlabel('CBF值')
    plt.ylabel('频率')
    plt.title('不安全样本的CBF值分布')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cbf_value_distribution.png'))
    plt.close()

def plot_feature_distribution(analysis_results, output_dir):
    """
    绘制高损失样本的特征分布
    """
    high_loss_safe = analysis_results['high_loss_safe_features']
    high_loss_unsafe = analysis_results['high_loss_unsafe_features']
    
    # 提取特征
    safe_tilt = [item['tilt'] for item in high_loss_safe]
    safe_rel_x = [item['rel_x'] for item in high_loss_safe]
    safe_rel_y = [item['rel_y'] for item in high_loss_safe]
    safe_rel_z = [item['rel_z'] for item in high_loss_safe]
    
    unsafe_tilt = [item['tilt'] for item in high_loss_unsafe]
    unsafe_rel_x = [item['rel_x'] for item in high_loss_unsafe]
    unsafe_rel_y = [item['rel_y'] for item in high_loss_unsafe]
    unsafe_rel_z = [item['rel_z'] for item in high_loss_unsafe]
    
    # 绘制特征散点图
    plt.figure(figsize=(15, 10))
    
    # Tilt vs Loss
    plt.subplot(2, 3, 1)
    plt.scatter([item['tilt'] for item in high_loss_safe], 
                [item['loss'] for item in high_loss_safe], 
                alpha=0.7, label='安全')
    plt.scatter([item['tilt'] for item in high_loss_unsafe], 
                [item['loss'] for item in high_loss_unsafe], 
                alpha=0.7, label='不安全')
    plt.xlabel('倾斜代价')
    plt.ylabel('损失')
    plt.title('倾斜vs损失')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # rel_x vs rel_y
    plt.subplot(2, 3, 2)
    plt.scatter(safe_rel_x, safe_rel_y, alpha=0.7, label='安全')
    plt.scatter(unsafe_rel_x, unsafe_rel_y, alpha=0.7, label='不安全')
    plt.xlabel('相对X')
    plt.ylabel('相对Y')
    plt.title('相对X vs Y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # rel_x vs rel_z
    plt.subplot(2, 3, 3)
    plt.scatter(safe_rel_x, safe_rel_z, alpha=0.7, label='安全')
    plt.scatter(unsafe_rel_x, unsafe_rel_z, alpha=0.7, label='不安全')
    plt.xlabel('相对X')
    plt.ylabel('相对Z')
    plt.title('相对X vs Z')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # rel_y vs rel_z
    plt.subplot(2, 3, 4)
    plt.scatter(safe_rel_y, safe_rel_z, alpha=0.7, label='安全')
    plt.scatter(unsafe_rel_y, unsafe_rel_z, alpha=0.7, label='不安全')
    plt.xlabel('相对Y')
    plt.ylabel('相对Z')
    plt.title('相对Y vs Z')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss vs CBF Value
    plt.subplot(2, 3, 5)
    plt.scatter([item['loss'] for item in high_loss_safe], 
                [item['cbf_val'] for item in high_loss_safe], 
                alpha=0.7, label='安全')
    plt.scatter([item['loss'] for item in high_loss_unsafe], 
                [item['cbf_val'] for item in high_loss_unsafe], 
                alpha=0.7, label='不安全')
    plt.xlabel('损失')
    plt.ylabel('CBF值')
    plt.title('损失vs CBF值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distribution.png'))
    plt.close()

def generate_report(analysis_results, output_dir):
    """
    生成分析报告
    """
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=============================================\n")
        f.write("NCBF模型评估报告\n")
        f.write("=============================================\n\n")
        
        f.write("安全样本损失统计:\n")
        for key, value in analysis_results['safe_stats'].items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")
        
        f.write("不安全样本损失统计:\n")
        for key, value in analysis_results['unsafe_stats'].items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")
        
        f.write("高损失安全样本的损失贡献分析:\n")
        f.write(f"  正号损失: {analysis_results['safe_contribution']['pos_sign']*100:.2f}%\n")
        f.write(f"  导数损失: {analysis_results['safe_contribution']['derivative']*100:.2f}%\n")
        f.write(f"  边界损失: {analysis_results['safe_contribution']['bounding']*100:.2f}%\n")
        f.write("\n")
        
        f.write("前5个最高损失安全样本:\n")
        for i, item in enumerate(analysis_results['top_safe_losses'][:5]):
            f.write(f"  样本 {i+1} (索引 {item['index']}):\n")
            f.write(f"    总损失: {item['total_loss']:.6f}\n")
            f.write(f"    正号损失: {item['pos_sign_loss']:.6f}\n")
            f.write(f"    导数损失: {item['derivative_loss']:.6f}\n")
            f.write(f"    边界损失: {item['bounding_loss']:.6f}\n")
            f.write(f"    CBF值: {item['cbf_val']:.6f}\n")
            f.write(f"    下一时刻CBF值: {item['cbf_next_val']:.6f}\n")
        f.write("\n")
        
        f.write("前5个最高损失不安全样本:\n")
        for i, item in enumerate(analysis_results['top_unsafe_losses'][:5]):
            f.write(f"  样本 {i+1} (索引 {item['index']}):\n")
            f.write(f"    总损失: {item['total_loss']:.6f}\n")
            f.write(f"    负号损失: {item['neg_sign_loss']:.6f}\n")
            f.write(f"    CBF值: {item['cbf_val']:.6f}\n")
        f.write("\n")
        
        # 分析高损失安全样本的特征模式
        f.write("高损失安全样本的特征模式:\n")
        safe_features = analysis_results['high_loss_safe_features']
        f.write(f"  平均倾斜代价: {np.mean([f['tilt'] for f in safe_features]):.4f}\n")
        f.write(f"  平均相对X: {np.mean([f['rel_x'] for f in safe_features]):.4f}\n")
        f.write(f"  平均相对Y: {np.mean([f['rel_y'] for f in safe_features]):.4f}\n")
        f.write(f"  平均相对Z: {np.mean([f['rel_z'] for f in safe_features]):.4f}\n")
        f.write(f"  平均CBF值: {np.mean([f['cbf_val'] for f in safe_features]):.4f}\n")
        f.write("\n")
        
        # 分析高损失不安全样本的特征模式
        f.write("高损失不安全样本的特征模式:\n")
        unsafe_features = analysis_results['high_loss_unsafe_features']
        f.write(f"  平均倾斜代价: {np.mean([f['tilt'] for f in unsafe_features]):.4f}\n")
        f.write(f"  平均相对X: {np.mean([f['rel_x'] for f in unsafe_features]):.4f}\n")
        f.write(f"  平均相对Y: {np.mean([f['rel_y'] for f in unsafe_features]):.4f}\n")
        f.write(f"  平均相对Z: {np.mean([f['rel_z'] for f in unsafe_features]):.4f}\n")
        f.write(f"  平均CBF值: {np.mean([f['cbf_val'] for f in unsafe_features]):.4f}\n")
        f.write("\n")
        
        f.write("=============================================\n")
        f.write("模型改进建议:\n")
        f.write("=============================================\n\n")
        
        # 基于分析结果给出建议
        if analysis_results['safe_contribution']['derivative'] > 0.5:
            f.write("- 考虑调整CBF_gamma参数，因为导数约束是损失的主要来源\n")
        
        if analysis_results['safe_contribution']['pos_sign'] > 0.5:
            f.write("- 考虑调整正样本的边际阈值(margin threshold)\n")
        
        if analysis_results['safe_contribution']['bounding'] > 0.5:
            f.write("- 考虑调整导数边界阈值(derivative bound threshold)\n")
        
        f.write("- 考虑添加更多与高损失模式相似的训练样本\n")
        f.write("- 如果高损失样本显示出时间模式，请检查模型架构，特别是序列处理部分\n")
        
    print(f"评估报告已保存到 {report_path}")

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载模型
    print(f"从{args.model}加载模型...")
    model = NCBF.load(
        args.model,
        hiddens=Hiddens,  # 从config.py导入
        seq_hiddens=SeqHiddens  # 从config.py导入
    )
    
    # 加载归一化器
    h_obj_normalizer, ee_normalizer = load_normalizers(args.h_obj_normalizer, args.ee_normalizer)
    
    # 加载数据
    data = load_formatted_data(args.data)
    safe_h_obj_xs, safe_h_obj_nxs, safe_ee_xs, safe_ee_nxs, unsafe_h_obj_xs, unsafe_ee_xs, original_data = data
    
    # 使用提供的helpers.py中的normalize_data函数进行归一化
    normalized_data = normalize_data(
        safe_h_obj_xs, safe_h_obj_nxs, unsafe_h_obj_xs,
        safe_ee_xs, safe_ee_nxs, unsafe_ee_xs,
        h_obj_normalizer, ee_normalizer, update=False
    )
    
    # 评估每个样本的损失
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    safe_losses, unsafe_losses = evaluate_individual_losses(
        model, normalized_data, device,
        margin_threshold=MarginThreshold,  # 从config.py导入
        derivative_threshold=DerivativeThreshold  # 从config.py导入
    )
    
    # 分析高损失样本
    analysis_results = analyze_high_loss_samples(
        safe_losses, unsafe_losses, args.top_n, original_data
    )
    
    # 绘制损失分布图
    plot_loss_distribution(safe_losses, unsafe_losses, args.output)
    
    # 绘制特征分布
    plot_feature_distribution(analysis_results, args.output)
    
    # 生成分析报告
    generate_report(analysis_results, args.output)
    
    # 保存详细结果
    with open(os.path.join(args.output, 'detailed_results.pickle'), 'wb') as f:
        pickle.dump({
            'safe_losses': safe_losses,
            'unsafe_losses': unsafe_losses,
            'analysis_results': analysis_results
        }, f)
    
    print(f"评估完成。结果已保存到 {args.output}")

if __name__ == "__main__":
    main()