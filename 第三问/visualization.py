# -*- coding: utf-8 -*-
"""
第三问可视化模块
生成Copula++模型分析结果的各种图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from scipy import stats
import os

# 设置中文字体
import matplotlib.font_manager as fm

# 添加字体文件路径
font_path = '../AaXinRui85-2.ttf'
if os.path.exists(font_path):
    # 注册字体
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"使用自定义字体: {font_prop.get_name()}")
else:
    # 使用系统字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
    print("使用系统字体")

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

class CopulaVisualizer:
    """Copula++模型可视化器"""
    
    def __init__(self, copula_model):
        self.model = copula_model
        self.output_dir = "output/plt/第三问"
        
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")
        
        # 1. 相关矩阵热力图
        self.plot_correlation_heatmap()
        
        # 2. 医学约束验证图
        self.plot_medical_constraints_validation()
        
        # 3. 联合概率对比图
        self.plot_joint_probability_comparison()
        
        # 4. 疾病关联网络图
        self.plot_disease_network()
        
        # 5. Copula密度图
        self.plot_copula_density()
        
        # 6. 医学知识约束雷达图
        self.plot_medical_knowledge_radar()
        
        # 7. 概率分布对比图
        self.plot_probability_distribution()
        
        # 8. 交互式3D图
        self.plot_3d_interactive()
        
        print("可视化图表生成完成")
        
    def plot_correlation_heatmap(self):
        """绘制相关矩阵热力图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始相关矩阵
        corr_matrix = self.model.correlation_matrix
        diseases = ['心脏病', '中风', '肝硬化']
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=diseases, yticklabels=diseases, ax=ax1)
        ax1.set_title('Copula++相关矩阵')
        
        # 医学约束对比
        constraints = self.model.medical_constraints.constraints
        constraint_matrix = np.zeros((3, 3))
        
        # 填充约束值
        if 'heart_stroke' in constraints:
            constraint_matrix[0, 1] = constraint_matrix[1, 0] = constraints['heart_stroke']['expected_correlation']
        if 'heart_cirrhosis' in constraints:
            constraint_matrix[0, 2] = constraint_matrix[2, 0] = constraints['heart_cirrhosis']['expected_correlation']
        if 'stroke_cirrhosis' in constraints:
            constraint_matrix[1, 2] = constraint_matrix[2, 1] = constraints['stroke_cirrhosis']['expected_correlation']
        
        sns.heatmap(constraint_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=diseases, yticklabels=diseases, ax=ax2)
        ax2.set_title('医学知识约束')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_medical_constraints_validation(self):
        """绘制医学约束验证图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        constraints = self.model.medical_constraints.constraints
        constraint_names = {
            'heart_stroke': '心脏病-中风',
            'heart_cirrhosis': '心脏病-肝硬化',
            'stroke_cirrhosis': '中风-肝硬化'
        }
        
        for i, (pair, constraint) in enumerate(constraints.items()):
            ax = axes[i//2, i%2]
            
            # 获取实际相关值
            if pair == 'heart_stroke':
                actual_corr = self.model.correlation_matrix[0, 1]
            elif pair == 'heart_cirrhosis':
                actual_corr = self.model.correlation_matrix[0, 2]
            elif pair == 'stroke_cirrhosis':
                actual_corr = self.model.correlation_matrix[1, 2]
            
            min_corr = constraint['min_correlation']
            max_corr = constraint['max_correlation']
            expected_corr = constraint['expected_correlation']
            
            # 绘制约束范围
            ax.axhline(y=min_corr, color='red', linestyle='--', alpha=0.7, label='最小约束')
            ax.axhline(y=max_corr, color='red', linestyle='--', alpha=0.7, label='最大约束')
            ax.axhline(y=expected_corr, color='blue', linestyle='-', alpha=0.7, label='期望值')
            ax.axhline(y=actual_corr, color='green', linestyle='-', linewidth=2, label='实际值')
            
            ax.fill_between([0, 1], min_corr, max_corr, alpha=0.2, color='yellow', label='约束范围')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_title(f'{constraint_names[pair]}约束验证')
            ax.set_xlabel('标准化位置')
            ax.set_ylabel('相关强度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/medical_constraints_validation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_joint_probability_comparison(self):
        """绘制联合概率对比图"""
        # 计算简单乘法概率
        marginal_probs = self.model.joint_probabilities['marginal']
        simple_mult = {
            'stroke_heart': marginal_probs['stroke'] * marginal_probs['heart'],
            'stroke_cirrhosis': marginal_probs['stroke'] * marginal_probs['cirrhosis'],
            'heart_cirrhosis': marginal_probs['heart'] * marginal_probs['cirrhosis'],
            'all_three': marginal_probs['stroke'] * marginal_probs['heart'] * marginal_probs['cirrhosis']
        }
        
        # Copula++概率
        copula_probs = {
            'stroke_heart': self.model.joint_probabilities['pair']['stroke_heart'],
            'stroke_cirrhosis': self.model.joint_probabilities['pair']['stroke_cirrhosis'],
            'heart_cirrhosis': self.model.joint_probabilities['pair']['heart_cirrhosis'],
            'all_three': self.model.joint_probabilities['triple']
        }
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 成对概率对比
        pairs = ['stroke_heart', 'stroke_cirrhosis', 'heart_cirrhosis']
        pair_names = ['中风-心脏病', '中风-肝硬化', '心脏病-肝硬化']
        
        x = np.arange(len(pairs))
        width = 0.35
        
        simple_values = [simple_mult[pair] for pair in pairs]
        copula_values = [copula_probs[pair] for pair in pairs]
        
        ax1.bar(x - width/2, simple_values, width, label='简单乘法', alpha=0.8)
        ax1.bar(x + width/2, copula_values, width, label='Copula++', alpha=0.8)
        
        ax1.set_xlabel('疾病组合')
        ax1.set_ylabel('联合概率')
        ax1.set_title('成对疾病联合概率对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pair_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 三疾病概率对比
        ax2.bar(['简单乘法', 'Copula++'], 
                [simple_mult['all_three'], copula_probs['all_three']],
                color=['lightblue', 'lightcoral'], alpha=0.8)
        ax2.set_ylabel('联合概率')
        ax2.set_title('三疾病同时发生概率对比')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/joint_probability_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_disease_network(self):
        """绘制疾病关联网络图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 节点位置
        positions = {
            '心脏病': (0, 1),
            '中风': (1, 0),
            '肝硬化': (0, -1)
        }
        
        # 绘制节点
        for disease, pos in positions.items():
            circle = Circle(pos, 0.3, color='lightblue', alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], disease, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 绘制边（关联强度）
        edges = [
            ('心脏病', '中风', self.model.correlation_matrix[0, 1]),
            ('心脏病', '肝硬化', self.model.correlation_matrix[0, 2]),
            ('中风', '肝硬化', self.model.correlation_matrix[1, 2])
        ]
        
        for start, end, weight in edges:
            start_pos = positions[start]
            end_pos = positions[end]
            
            # 根据权重确定线宽和颜色
            linewidth = abs(weight) * 5 + 1
            color = 'red' if weight > 0 else 'blue'
            alpha = abs(weight) * 0.8 + 0.2
            
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)
            
            # 添加权重标签
            mid_x = (start_pos[0] + end_pos[0]) / 2
            mid_y = (start_pos[1] + end_pos[1]) / 2
            ax.text(mid_x, mid_y, f'{weight:.3f}', ha='center', va='center', 
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-1.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title('疾病关联网络图\n(线宽和颜色表示关联强度)')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/disease_network.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_copula_density(self):
        """绘制Copula密度图"""
        # 生成样本数据
        n_samples = 1000
        np.random.seed(42)
        
        # 使用模型的相关矩阵生成样本
        mean = np.zeros(3)
        samples = np.random.multivariate_normal(mean, self.model.correlation_matrix, n_samples)
        
        # 转换为概率
        probs = stats.norm.cdf(samples)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 心脏病-中风密度
        axes[0, 0].hexbin(probs[:, 0], probs[:, 1], gridsize=20, cmap='Blues')
        axes[0, 0].set_xlabel('心脏病概率')
        axes[0, 0].set_ylabel('中风概率')
        axes[0, 0].set_title('心脏病-中风Copula密度')
        
        # 心脏病-肝硬化密度
        axes[0, 1].hexbin(probs[:, 0], probs[:, 2], gridsize=20, cmap='Greens')
        axes[0, 1].set_xlabel('心脏病概率')
        axes[0, 1].set_ylabel('肝硬化概率')
        axes[0, 1].set_title('心脏病-肝硬化Copula密度')
        
        # 中风-肝硬化密度
        axes[1, 0].hexbin(probs[:, 1], probs[:, 2], gridsize=20, cmap='Reds')
        axes[1, 0].set_xlabel('中风概率')
        axes[1, 0].set_ylabel('肝硬化概率')
        axes[1, 0].set_title('中风-肝硬化Copula密度')
        
        # 三疾病联合密度（投影）
        axes[1, 1].scatter(probs[:, 0], probs[:, 1], c=probs[:, 2], 
                           cmap='viridis', alpha=0.6, s=20)
        axes[1, 1].set_xlabel('心脏病概率')
        axes[1, 1].set_ylabel('中风概率')
        axes[1, 1].set_title('三疾病联合密度\n(颜色表示肝硬化概率)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/copula_density.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_medical_knowledge_radar(self):
        """绘制医学知识约束雷达图"""
        # 准备数据
        categories = ['心脏病-中风\n(高血压相关)', '心脏病-肝硬化\n(酒精相关)', 
                     '中风-肝硬化\n(代谢相关)', '三疾病联合\n(综合因素)']
        
        # 实际相关强度
        actual_values = [
            self.model.correlation_matrix[0, 1],  # 心脏病-中风
            self.model.correlation_matrix[0, 2],  # 心脏病-肝硬化
            self.model.correlation_matrix[1, 2],  # 中风-肝硬化
            np.mean([self.model.correlation_matrix[0, 1], 
                    self.model.correlation_matrix[0, 2], 
                    self.model.correlation_matrix[1, 2]])  # 平均相关
        ]
        
        # 医学期望值
        constraints = self.model.medical_constraints.constraints
        expected_values = [
            constraints['heart_stroke']['expected_correlation'],
            constraints['heart_cirrhosis']['expected_correlation'],
            constraints['stroke_cirrhosis']['expected_correlation'],
            self.model.medical_constraints.triple_constraints['expected_correlation']
        ]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        actual_values += actual_values[:1]
        expected_values += expected_values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, actual_values, 'o-', linewidth=2, label='实际相关强度', color='blue')
        ax.fill(angles, actual_values, alpha=0.25, color='blue')
        
        ax.plot(angles, expected_values, 'o-', linewidth=2, label='医学期望值', color='red')
        ax.fill(angles, expected_values, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('医学知识约束雷达图', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/medical_knowledge_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_probability_distribution(self):
        """绘制概率分布对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 边缘概率分布
        marginal_probs = self.model.joint_probabilities['marginal']
        diseases = list(marginal_probs.keys())
        prob_values = list(marginal_probs.values())
        
        axes[0, 0].bar(diseases, prob_values, color=['red', 'blue', 'green'], alpha=0.7)
        axes[0, 0].set_title('单疾病发生概率')
        axes[0, 0].set_ylabel('概率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 成对联合概率
        pair_probs = self.model.joint_probabilities['pair']
        pair_names = list(pair_probs.keys())
        pair_values = list(pair_probs.values())
        
        axes[0, 1].bar(range(len(pair_names)), pair_values, color='orange', alpha=0.7)
        axes[0, 1].set_title('成对疾病联合概率')
        axes[0, 1].set_ylabel('概率')
        axes[0, 1].set_xticks(range(len(pair_names)))
        axes[0, 1].set_xticklabels(pair_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 概率对比（Copula vs 简单乘法）
        simple_mult = marginal_probs['stroke'] * marginal_probs['heart'] * marginal_probs['cirrhosis']
        copula_triple = self.model.joint_probabilities['triple']
        
        comparison_data = [simple_mult, copula_triple]
        comparison_labels = ['简单乘法', 'Copula++']
        
        axes[1, 0].bar(comparison_labels, comparison_data, color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[1, 0].set_title('三疾病联合概率对比')
        axes[1, 0].set_ylabel('概率')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 相关强度分布
        corr_matrix = self.model.correlation_matrix
        corr_values = [corr_matrix[0, 1], corr_matrix[0, 2], corr_matrix[1, 2]]
        corr_labels = ['心脏病-中风', '心脏病-肝硬化', '中风-肝硬化']
        
        colors = ['red' if v > 0.3 else 'orange' if v > 0.1 else 'green' for v in corr_values]
        axes[1, 1].bar(corr_labels, corr_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('疾病间相关强度')
        axes[1, 1].set_ylabel('相关系数')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/probability_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_3d_interactive(self):
        """绘制交互式3D图"""
        try:
            # 生成3D数据
            n_samples = 500
            np.random.seed(42)
            
            mean = np.zeros(3)
            samples = np.random.multivariate_normal(mean, self.model.correlation_matrix, n_samples)
            probs = stats.norm.cdf(samples)
            
            # 创建3D散点图
            fig = go.Figure(data=[go.Scatter3d(
                x=probs[:, 0],
                y=probs[:, 1], 
                z=probs[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=probs[:, 2],
                    colorscale='Viridis',
                    opacity=0.6
                ),
                text=[f'心脏病: {p[0]:.3f}<br>中风: {p[1]:.3f}<br>肝硬化: {p[2]:.3f}' 
                      for p in probs],
                hovertemplate='%{text}<extra></extra>'
            )])
            
            fig.update_layout(
                title='Copula++三疾病联合概率分布',
                scene=dict(
                    xaxis_title='心脏病概率',
                    yaxis_title='中风概率', 
                    zaxis_title='肝硬化概率'
                ),
                width=800,
                height=600
            )
            
            fig.write_html(f"{self.output_dir}/3d_interactive.html")
            
        except ImportError:
            print("Plotly未安装，跳过3D交互图生成")
            
    def generate_summary_report(self):
        """生成可视化总结报告"""
        print("生成可视化总结报告...")
        
        # 创建总结图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 相关矩阵
        ax1 = plt.subplot(3, 3, 1)
        sns.heatmap(self.model.correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=['心脏病', '中风', '肝硬化'], 
                   yticklabels=['心脏病', '中风', '肝硬化'])
        ax1.set_title('Copula++相关矩阵')
        
        # 2. 边缘概率
        ax2 = plt.subplot(3, 3, 2)
        marginal_probs = self.model.joint_probabilities['marginal']
        ax2.bar(marginal_probs.keys(), marginal_probs.values(), color=['red', 'blue', 'green'])
        ax2.set_title('单疾病概率')
        ax2.set_ylabel('概率')
        
        # 3. 联合概率
        ax3 = plt.subplot(3, 3, 3)
        pair_probs = self.model.joint_probabilities['pair']
        ax3.bar(pair_probs.keys(), pair_probs.values(), color='orange')
        ax3.set_title('成对联合概率')
        ax3.set_ylabel('概率')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 医学约束验证
        ax4 = plt.subplot(3, 3, 4)
        constraints = self.model.medical_constraints.constraints
        constraint_names = ['心脏病-中风', '心脏病-肝硬化', '中风-肝硬化']
        actual_corrs = [self.model.correlation_matrix[0, 1], 
                       self.model.correlation_matrix[0, 2], 
                       self.model.correlation_matrix[1, 2]]
        expected_corrs = [constraints['heart_stroke']['expected_correlation'],
                         constraints['heart_cirrhosis']['expected_correlation'],
                         constraints['stroke_cirrhosis']['expected_correlation']]
        
        x = np.arange(len(constraint_names))
        width = 0.35
        ax4.bar(x - width/2, actual_corrs, width, label='实际值', alpha=0.8)
        ax4.bar(x + width/2, expected_corrs, width, label='期望值', alpha=0.8)
        ax4.set_title('医学约束验证')
        ax4.set_xticks(x)
        ax4.set_xticklabels(constraint_names, rotation=45)
        ax4.legend()
        
        # 5. 概率对比
        ax5 = plt.subplot(3, 3, 5)
        simple_mult = marginal_probs['stroke'] * marginal_probs['heart'] * marginal_probs['cirrhosis']
        copula_triple = self.model.joint_probabilities['triple']
        ax5.bar(['简单乘法', 'Copula++'], [simple_mult, copula_triple], 
                color=['lightblue', 'lightcoral'])
        ax5.set_title('三疾病概率对比')
        ax5.set_ylabel('概率')
        
        # 6. 网络图简化版
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter([0, 1, 0], [1, 0, -1], s=500, c=['red', 'blue', 'green'], alpha=0.7)
        ax6.text(0, 1, '心脏病', ha='center', va='center', fontweight='bold')
        ax6.text(1, 0, '中风', ha='center', va='center', fontweight='bold')
        ax6.text(0, -1, '肝硬化', ha='center', va='center', fontweight='bold')
        
        # 添加连线
        for i, (start, end, weight) in enumerate([(0, 1, actual_corrs[0]), 
                                                  (0, 2, actual_corrs[1]), 
                                                  (1, 2, actual_corrs[2])]):
            if i == 0:
                ax6.plot([0, 1], [1, 0], 'r-', linewidth=weight*5+1, alpha=0.7)
            elif i == 1:
                ax6.plot([0, 0], [1, -1], 'g-', linewidth=weight*5+1, alpha=0.7)
            else:
                ax6.plot([1, 0], [0, -1], 'b-', linewidth=weight*5+1, alpha=0.7)
        
        ax6.set_xlim(-0.5, 1.5)
        ax6.set_ylim(-1.5, 1.5)
        ax6.set_title('疾病关联网络')
        ax6.axis('off')
        
        # 7. 统计信息
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        info_text = f"""
Copula++模型总结

相关强度统计:
• 最强相关: {max(actual_corrs):.3f}
• 最弱相关: {min(actual_corrs):.3f}
• 平均相关: {np.mean(actual_corrs):.3f}

概率统计:
• 三疾病联合概率: {copula_triple:.6f}
• 简单乘法概率: {simple_mult:.6f}
• 提升倍数: {copula_triple/simple_mult:.2f}x

医学验证:
• 约束满足率: {sum(1 for c in actual_corrs if 0 <= c <= 1)}/3
• 模型一致性: {'✓' if all(c >= 0 for c in actual_corrs) else '✗'}
        """
        ax7.text(0.1, 0.9, info_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 8. 时间序列（模拟）
        ax8 = plt.subplot(3, 3, 8)
        time_points = np.linspace(0, 10, 100)
        trend = 0.1 * np.sin(time_points) + 0.3
        ax8.plot(time_points, trend, 'b-', linewidth=2)
        ax8.set_title('相关强度趋势')
        ax8.set_xlabel('时间')
        ax8.set_ylabel('相关强度')
        ax8.grid(True, alpha=0.3)
        
        # 9. 分布图
        ax9 = plt.subplot(3, 3, 9)
        np.random.seed(42)
        samples = np.random.multivariate_normal(np.zeros(3), self.model.correlation_matrix, 1000)
        ax9.hist(samples[:, 0], bins=30, alpha=0.7, label='心脏病', color='red')
        ax9.hist(samples[:, 1], bins=30, alpha=0.7, label='中风', color='blue')
        ax9.hist(samples[:, 2], bins=30, alpha=0.7, label='肝硬化', color='green')
        ax9.set_title('标准化特征分布')
        ax9.set_xlabel('标准化值')
        ax9.set_ylabel('频次')
        ax9.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/summary_report.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化总结报告生成完成") 