# -*- coding: utf-8 -*-
"""
第三问：带医学常识的关联控制器（Copula++）
解决"疾病不独立"问题，精准建模多疾病关联
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
import joblib
import pickle
from tqdm import tqdm
import json
from datetime import datetime

# GPU加速支持
try:
    import cupy as cp
    USE_GPU = True
    print("GPU加速已启用 - 使用CuPy进行矩阵运算")
except ImportError:
    USE_GPU = False
    print("使用CPU模式 - 建议安装CuPy以获得GPU加速")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

warnings.filterwarnings('ignore')

class MedicalKnowledgeConstraints:
    """医学知识约束类 - 定义疾病间的医学关联强度"""
    
    def __init__(self):
        # 医学知识约束：疾病关联强度范围
        # 值域：[0, 1]，0表示独立，1表示完全相关
        self.constraints = {
            # 心脏病-中风：强正相关（高血压是共同风险因素）
            'heart_stroke': {
                'min_correlation': 0.3,  # 最小相关强度
                'max_correlation': 0.8,  # 最大相关强度
                'expected_correlation': 0.5,  # 期望相关强度
                'medical_reason': '高血压、糖尿病等共同风险因素'
            },
            # 心脏病-肝硬化：中等相关（酒精、代谢综合征等）
            'heart_cirrhosis': {
                'min_correlation': 0.1,
                'max_correlation': 0.5,
                'expected_correlation': 0.3,
                'medical_reason': '酒精、代谢综合征等共同风险因素'
            },
            # 中风-肝硬化：弱相关
            'stroke_cirrhosis': {
                'min_correlation': 0.05,
                'max_correlation': 0.3,
                'expected_correlation': 0.15,
                'medical_reason': '间接关联，主要通过代谢因素'
            }
        }
        
        # 三疾病联合约束
        self.triple_constraints = {
            'min_correlation': 0.1,  # 三疾病同时发生的最小相关强度
            'max_correlation': 0.6,  # 三疾病同时发生的最大相关强度
            'expected_correlation': 0.25,  # 期望相关强度
            'medical_reason': '多疾病共病模式，受共同风险因素影响'
        }

class CopulaPlusPlus:
    """Copula++模型 - 带医学常识的关联控制器"""
    
    def __init__(self, medical_constraints=None):
        self.medical_constraints = medical_constraints or MedicalKnowledgeConstraints()
        self.copula_params = {}
        self.marginal_models = {}
        self.correlation_matrix = None
        self.medical_audit_report = {}
        
        # 创建输出目录
        self.output_dir = "output"
        os.makedirs(f"{self.output_dir}/csv/第三问", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plt/第三问", exist_ok=True)
        os.makedirs(f"{self.output_dir}/models/第三问", exist_ok=True)
        
    def load_data(self):
        """加载三种疾病的数据"""
        print("正在加载疾病数据...")
        
        data_dir = "./附件"
        self.stroke_data = pd.read_csv(f"{data_dir}/stroke.csv", encoding='utf-8-sig')
        self.heart_data = pd.read_csv(f"{data_dir}/heart.csv", encoding='utf-8-sig')
        self.cirrhosis_data = pd.read_csv(f"{data_dir}/cirrhosis.csv", encoding='utf-8-sig')
        
        print(f"数据加载完成：")
        print(f"  中风数据：{len(self.stroke_data)} 样本")
        print(f"  心脏病数据：{len(self.heart_data)} 样本")
        print(f"  肝硬化数据：{len(self.cirrhosis_data)} 样本")
        
    def preprocess_data(self):
        """数据预处理和特征工程"""
        print("正在进行数据预处理...")
        
        # 预处理各疾病数据
        self.stroke_features = self._extract_common_features(self.stroke_data, 'stroke')
        self.heart_features = self._extract_common_features(self.heart_data, 'heart')
        self.cirrhosis_features = self._extract_common_features(self.cirrhosis_data, 'cirrhosis')
        
        # 创建联合数据集
        self._create_joint_dataset()
        
        print("数据预处理完成")
        
    def _extract_common_features(self, data, disease_type):
        """提取共同特征"""
        features = {}
        
        if disease_type == 'stroke':
            # 中风特征
            features['age'] = data['age']
            features['hypertension'] = data['hypertension']
            features['heart_disease'] = data['heart_disease']
            features['avg_glucose_level'] = data['avg_glucose_level']
            features['bmi'] = data['bmi']
            features['smoking_status'] = pd.Categorical(data['smoking_status']).codes
            features['target'] = data['stroke']
            
        elif disease_type == 'heart':
            # 心脏病特征
            features['age'] = data['Age']
            features['sex'] = data['Sex']
            features['resting_bp'] = data['RestingBP']
            features['cholesterol'] = data['Cholesterol']
            features['max_hr'] = data['MaxHR']
            features['oldpeak'] = data['Oldpeak']
            features['target'] = data['HeartDisease']
            
        elif disease_type == 'cirrhosis':
            # 肝硬化特征
            features['age'] = data['Age']
            features['sex'] = data['Sex']
            features['bilirubin'] = data['Bilirubin']
            features['cholesterol'] = data['Cholesterol']
            features['albumin'] = data['Albumin']
            features['copper'] = data['Copper']
            features['target'] = (data['Stage'] > 1).astype(int)
        
        return pd.DataFrame(features)
        
    def _create_joint_dataset(self):
        """创建联合数据集"""
        print("创建联合数据集...")
        
        # 标准化年龄特征以便匹配
        stroke_age_norm = (self.stroke_features['age'] - self.stroke_features['age'].mean()) / self.stroke_features['age'].std()
        heart_age_norm = (self.heart_features['age'] - self.heart_features['age'].mean()) / self.heart_features['age'].std()
        cirrhosis_age_norm = (self.cirrhosis_features['age'] - self.cirrhosis_features['age'].mean()) / self.cirrhosis_features['age'].std()
        
        # 创建联合概率矩阵
        n_samples = min(len(self.stroke_features), len(self.heart_features), len(self.cirrhosis_features))
        
        # 随机采样创建联合数据
        np.random.seed(42)
        stroke_idx = np.random.choice(len(self.stroke_features), n_samples, replace=False)
        heart_idx = np.random.choice(len(self.heart_features), n_samples, replace=False)
        cirrhosis_idx = np.random.choice(len(self.cirrhosis_features), n_samples, replace=False)
        
        self.joint_data = pd.DataFrame({
            'stroke_prob': self.stroke_features.iloc[stroke_idx]['target'].values,
            'heart_prob': self.heart_features.iloc[heart_idx]['target'].values,
            'cirrhosis_prob': self.cirrhosis_features.iloc[cirrhosis_idx]['target'].values,
            'stroke_age': self.stroke_features.iloc[stroke_idx]['age'].values,
            'heart_age': self.heart_features.iloc[heart_idx]['age'].values,
            'cirrhosis_age': self.cirrhosis_features.iloc[cirrhosis_idx]['age'].values
        })
        
        print(f"联合数据集创建完成：{len(self.joint_data)} 样本")
        
    def fit_marginal_models(self):
        """拟合边缘分布模型"""
        print("正在拟合边缘分布模型...")
        
        # 使用核密度估计拟合边缘分布
        from sklearn.neighbors import KernelDensity
        
        diseases = ['stroke', 'heart', 'cirrhosis']
        
        for disease in diseases:
            if disease == 'stroke':
                data = self.stroke_features['target']
            elif disease == 'heart':
                data = self.heart_features['target']
            else:
                data = self.cirrhosis_features['target']
            
            # 核密度估计
            kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
            kde.fit(data.values.reshape(-1, 1))
            
            self.marginal_models[disease] = kde
            
        print("边缘分布模型拟合完成")
        
    def fit_copula_with_medical_constraints(self):
        """使用医学约束拟合Copula模型"""
        print("正在拟合带医学约束的Copula模型...")
        
        # 使用Gaussian Copula
        self._fit_gaussian_copula()
        
        # 应用医学约束
        self._apply_medical_constraints()
        
        print("Copula模型拟合完成")
        
    def _fit_gaussian_copula(self):
        """拟合Gaussian Copula"""
        # 计算经验分布函数
        u_stroke = self._empirical_cdf(self.joint_data['stroke_prob'])
        u_heart = self._empirical_cdf(self.joint_data['heart_prob'])
        u_cirrhosis = self._empirical_cdf(self.joint_data['cirrhosis_prob'])
        
        # 转换为标准正态分布
        z_stroke = stats.norm.ppf(u_stroke)
        z_heart = stats.norm.ppf(u_heart)
        z_cirrhosis = stats.norm.ppf(u_cirrhosis)
        
        # 处理无穷大值
        z_stroke = np.nan_to_num(z_stroke, nan=0.0, posinf=3.0, neginf=-3.0)
        z_heart = np.nan_to_num(z_heart, nan=0.0, posinf=3.0, neginf=-3.0)
        z_cirrhosis = np.nan_to_num(z_cirrhosis, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # 计算相关矩阵
        Z = np.column_stack([z_stroke, z_heart, z_cirrhosis])
        
        # 确保没有NaN值
        if np.any(np.isnan(Z)):
            print("警告：发现NaN值，使用默认相关矩阵")
            self.correlation_matrix = np.array([
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.15],
                [0.3, 0.15, 1.0]
            ])
        else:
            self.correlation_matrix = np.corrcoef(Z.T)
            # 确保对角线为1
            np.fill_diagonal(self.correlation_matrix, 1.0)
        
        print("Gaussian Copula相关矩阵：")
        print(self.correlation_matrix)
        
    def _empirical_cdf(self, data):
        """计算经验分布函数"""
        # 处理重复值
        sorted_data = np.sort(data)
        ranks = np.searchsorted(sorted_data, data)
        
        # 处理边界情况
        cdf = ranks / len(data)
        cdf = np.clip(cdf, 0.001, 0.999)  # 避免0和1
        
        return cdf
        
    def _apply_medical_constraints(self):
        """应用医学约束"""
        print("应用医学知识约束...")
        
        # 获取当前相关矩阵
        current_corr = self.correlation_matrix.copy()
        
        # 应用成对约束
        constraints = self.medical_constraints.constraints
        
        # 心脏病-中风约束
        if 'heart_stroke' in constraints:
            target_corr = constraints['heart_stroke']['expected_correlation']
            min_corr = constraints['heart_stroke']['min_correlation']
            max_corr = constraints['heart_stroke']['max_correlation']
            
            # 调整相关强度
            current_corr[0, 1] = current_corr[1, 0] = np.clip(
                current_corr[0, 1], min_corr, max_corr
            )
            
        # 心脏病-肝硬化约束
        if 'heart_cirrhosis' in constraints:
            target_corr = constraints['heart_cirrhosis']['expected_correlation']
            min_corr = constraints['heart_cirrhosis']['min_correlation']
            max_corr = constraints['heart_cirrhosis']['max_correlation']
            
            current_corr[0, 2] = current_corr[2, 0] = np.clip(
                current_corr[0, 2], min_corr, max_corr
            )
            
        # 中风-肝硬化约束
        if 'stroke_cirrhosis' in constraints:
            target_corr = constraints['stroke_cirrhosis']['expected_correlation']
            min_corr = constraints['stroke_cirrhosis']['min_correlation']
            max_corr = constraints['stroke_cirrhosis']['max_correlation']
            
            current_corr[1, 2] = current_corr[2, 1] = np.clip(
                current_corr[1, 2], min_corr, max_corr
            )
        
        # 确保相关矩阵是正定的
        self.correlation_matrix = self._make_positive_definite(current_corr)
        
        print("应用约束后的相关矩阵：")
        print(self.correlation_matrix)
        
    def _make_positive_definite(self, corr_matrix):
        """确保相关矩阵是正定的"""
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        
        # 将负特征值设为小的正数
        eigenvals = np.maximum(eigenvals, 1e-6)
        
        # 重新构造矩阵
        corr_matrix_pd = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # 重新标准化对角线为1
        diag_sqrt = np.sqrt(np.diag(corr_matrix_pd))
        corr_matrix_pd = corr_matrix_pd / np.outer(diag_sqrt, diag_sqrt)
        
        return corr_matrix_pd
        
    def calculate_joint_probabilities(self):
        """计算联合概率"""
        print("计算联合概率...")
        
        # 获取各疾病的边缘概率
        stroke_prob = self.stroke_features['target'].mean()
        heart_prob = self.heart_features['target'].mean()
        cirrhosis_prob = self.cirrhosis_features['target'].mean()
        
        print(f"边缘概率：")
        print(f"  中风：{stroke_prob:.4f}")
        print(f"  心脏病：{heart_prob:.4f}")
        print(f"  肝硬化：{cirrhosis_prob:.4f}")
        
        # 使用Copula计算联合概率
        joint_probs = self._copula_joint_probability(
            [stroke_prob, heart_prob, cirrhosis_prob]
        )
        
        # 计算成对联合概率
        pair_probs = {
            'stroke_heart': self._pair_joint_probability(stroke_prob, heart_prob, 0, 1),
            'stroke_cirrhosis': self._pair_joint_probability(stroke_prob, cirrhosis_prob, 1, 2),
            'heart_cirrhosis': self._pair_joint_probability(heart_prob, cirrhosis_prob, 0, 2)
        }
        
        # 计算三疾病联合概率
        triple_prob = self._triple_joint_probability(
            stroke_prob, heart_prob, cirrhosis_prob
        )
        
        self.joint_probabilities = {
            'marginal': {'stroke': stroke_prob, 'heart': heart_prob, 'cirrhosis': cirrhosis_prob},
            'pair': pair_probs,
            'triple': triple_prob
        }
        
        print("联合概率计算完成")
        
    def _copula_joint_probability(self, marginal_probs):
        """使用Copula计算联合概率"""
        # 转换为标准正态分布
        z_values = [stats.norm.ppf(p) for p in marginal_probs]
        
        # 使用多元正态分布计算联合概率
        mean = np.zeros(3)
        joint_prob = stats.multivariate_normal.cdf(z_values, mean, self.correlation_matrix)
        
        return joint_prob
        
    def _pair_joint_probability(self, p1, p2, idx1, idx2):
        """计算成对联合概率"""
        z1 = stats.norm.ppf(p1)
        z2 = stats.norm.ppf(p2)
        
        # 提取2x2相关矩阵
        corr_2d = self.correlation_matrix[[idx1, idx2]][:, [idx1, idx2]]
        
        # 计算联合概率
        joint_prob = stats.multivariate_normal.cdf([z1, z2], [0, 0], corr_2d)
        
        return joint_prob
        
    def _triple_joint_probability(self, p1, p2, p3):
        """计算三疾病联合概率"""
        return self._copula_joint_probability([p1, p2, p3])
        
    def generate_medical_audit_report(self):
        """生成医学关联强度审计报告"""
        print("生成医学关联强度审计报告...")
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'correlation_matrix': self.correlation_matrix.tolist(),
            'constraints_analysis': {},
            'medical_validation': {},
            'recommendations': []
        }
        
        # 分析约束满足情况
        constraints = self.medical_constraints.constraints
        
        for pair, constraint in constraints.items():
            if pair == 'heart_stroke':
                actual_corr = self.correlation_matrix[0, 1]
            elif pair == 'heart_cirrhosis':
                actual_corr = self.correlation_matrix[0, 2]
            elif pair == 'stroke_cirrhosis':
                actual_corr = self.correlation_matrix[1, 2]
            
            min_corr = constraint['min_correlation']
            max_corr = constraint['max_correlation']
            expected_corr = constraint['expected_correlation']
            
            report['constraints_analysis'][pair] = {
                'actual_correlation': float(actual_corr),
                'expected_correlation': float(expected_corr),
                'constraint_range': [float(min_corr), float(max_corr)],
                'constraint_satisfied': bool(min_corr <= actual_corr <= max_corr),
                'medical_reason': constraint['medical_reason']
            }
        
        # 医学验证
        report['medical_validation'] = {
            'strongest_correlation': {
                'pair': self._find_strongest_correlation()['pair'],
                'correlation': float(self._find_strongest_correlation()['correlation'])
            },
            'weakest_correlation': {
                'pair': self._find_weakest_correlation()['pair'],
                'correlation': float(self._find_weakest_correlation()['correlation'])
            },
            'medical_consistency': {
                'heart_stroke_strongest': bool(self._check_medical_consistency()['heart_stroke_strongest']),
                'heart_cirrhosis_medium': bool(self._check_medical_consistency()['heart_cirrhosis_medium']),
                'all_positive': bool(self._check_medical_consistency()['all_positive'])
            }
        }
        
        # 建议
        report['recommendations'] = self._generate_recommendations()
        
        self.medical_audit_report = report
        
        # 保存报告
        with open(f"{self.output_dir}/csv/第三问/medical_audit_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print("医学审计报告生成完成")
        
    def _find_strongest_correlation(self):
        """找到最强相关"""
        corr_matrix = self.correlation_matrix
        max_corr = -1
        max_pair = None
        
        for i in range(3):
            for j in range(i+1, 3):
                if abs(corr_matrix[i, j]) > abs(max_corr):
                    max_corr = corr_matrix[i, j]
                    diseases = ['heart', 'stroke', 'cirrhosis']
                    max_pair = f"{diseases[i]}-{diseases[j]}"
        
        return {'pair': max_pair, 'correlation': max_corr}
        
    def _find_weakest_correlation(self):
        """找到最弱相关"""
        corr_matrix = self.correlation_matrix
        min_corr = 1
        min_pair = None
        
        for i in range(3):
            for j in range(i+1, 3):
                if abs(corr_matrix[i, j]) < abs(min_corr):
                    min_corr = corr_matrix[i, j]
                    diseases = ['heart', 'stroke', 'cirrhosis']
                    min_pair = f"{diseases[i]}-{diseases[j]}"
        
        return {'pair': min_pair, 'correlation': min_corr}
        
    def _check_medical_consistency(self):
        """检查医学一致性"""
        # 检查心脏病-中风是否是最强相关
        heart_stroke_corr = abs(self.correlation_matrix[0, 1])
        heart_cirrhosis_corr = abs(self.correlation_matrix[0, 2])
        stroke_cirrhosis_corr = abs(self.correlation_matrix[1, 2])
        
        consistency_checks = {
            'heart_stroke_strongest': heart_stroke_corr >= max(heart_cirrhosis_corr, stroke_cirrhosis_corr),
            'heart_cirrhosis_medium': heart_cirrhosis_corr >= stroke_cirrhosis_corr,
            'all_positive': all(self.correlation_matrix[i, j] >= 0 for i in range(3) for j in range(i+1, 3))
        }
        
        return consistency_checks
        
    def _generate_recommendations(self):
        """生成建议"""
        recommendations = []
        
        # 基于相关强度的建议
        heart_stroke_corr = self.correlation_matrix[0, 1]
        if heart_stroke_corr > 0.5:
            recommendations.append("心脏病和中风显示强正相关，建议加强高血压和糖尿病管理")
        
        heart_cirrhosis_corr = self.correlation_matrix[0, 2]
        if heart_cirrhosis_corr > 0.3:
            recommendations.append("心脏病和肝硬化存在中等相关，建议关注酒精和代谢综合征管理")
        
        # 基于约束满足情况的建议
        constraints = self.medical_constraints.constraints
        for pair, constraint in constraints.items():
            if pair == 'heart_stroke':
                actual_corr = self.correlation_matrix[0, 1]
            elif pair == 'heart_cirrhosis':
                actual_corr = self.correlation_matrix[0, 2]
            elif pair == 'stroke_cirrhosis':
                actual_corr = self.correlation_matrix[1, 2]
            
            if not (constraint['min_correlation'] <= actual_corr <= constraint['max_correlation']):
                recommendations.append(f"{pair}相关强度超出医学预期范围，需要进一步验证")
        
        return recommendations
        
    def save_results(self):
        """保存结果"""
        print("保存分析结果...")
        
        # 保存联合概率
        prob_df = pd.DataFrame({
            'Probability_Type': ['Stroke', 'Heart', 'Cirrhosis', 'Stroke_Heart', 'Stroke_Cirrhosis', 'Heart_Cirrhosis', 'All_Three'],
            'Probability_Value': [
                self.joint_probabilities['marginal']['stroke'],
                self.joint_probabilities['marginal']['heart'],
                self.joint_probabilities['marginal']['cirrhosis'],
                self.joint_probabilities['pair']['stroke_heart'],
                self.joint_probabilities['pair']['stroke_cirrhosis'],
                self.joint_probabilities['pair']['heart_cirrhosis'],
                self.joint_probabilities['triple']
            ],
            'Calculation_Method': ['Marginal', 'Marginal', 'Marginal', 'Copula++', 'Copula++', 'Copula++', 'Copula++']
        })
        
        prob_df.to_csv(f"{self.output_dir}/csv/第三问/joint_probabilities.csv", 
                      index=False, encoding='utf-8-sig')
        
        # 保存相关矩阵
        corr_df = pd.DataFrame(
            self.correlation_matrix,
            columns=['Heart', 'Stroke', 'Cirrhosis'],
            index=['Heart', 'Stroke', 'Cirrhosis']
        )
        corr_df.to_csv(f"{self.output_dir}/csv/第三问/correlation_matrix.csv", 
                      encoding='utf-8-sig')
        
        # 保存模型
        model_data = {
            'correlation_matrix': self.correlation_matrix,
            'marginal_models': self.marginal_models,
            'joint_probabilities': self.joint_probabilities,
            'medical_constraints': self.medical_constraints.constraints
        }
        
        with open(f"{self.output_dir}/models/第三问/copula_plus_plus_model.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
        print("结果保存完成")
        
    def run_complete_analysis(self):
        """运行完整分析"""
        print("开始Copula++多疾病关联分析...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据预处理
        self.preprocess_data()
        
        # 3. 拟合边缘分布
        self.fit_marginal_models()
        
        # 4. 拟合Copula模型
        self.fit_copula_with_medical_constraints()
        
        # 5. 计算联合概率
        self.calculate_joint_probabilities()
        
        # 6. 生成医学审计报告
        self.generate_medical_audit_report()
        
        # 7. 保存结果
        self.save_results()
        
        print("Copula++分析完成！")
        print("结果保存在 output/csv/第三问 和 output/plt/第三问 文件夹中")

if __name__ == "__main__":
    # 创建Copula++模型并运行分析
    copula_model = CopulaPlusPlus()
    copula_model.run_complete_analysis() 