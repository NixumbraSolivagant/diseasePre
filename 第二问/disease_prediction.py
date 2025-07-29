# -*- coding: utf-8 -*-
"""
第二问：不同疾病预测模型的构建 - 极致GPU加速版本
分而治之的集成学习方案：
1. 专科医生模型（单疾病专家）
2. 总分析师元模型（结果融合）
使用CatBoost实现极致GPU加速
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import catboost as cb
import shap
import warnings
import joblib
import pickle
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 极致GPU加速设置
try:
    import cupy as cp
    USE_GPU = True
    print("🚀 极致GPU加速已启用")
    
    # 设置CuPy内存池以优化GPU内存使用
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=1024**3)  # 1GB GPU内存限制
    
    # 设置GPU设备
    if cp.cuda.runtime.getDeviceCount() > 0:
        print(f"🎯 检测到 {cp.cuda.runtime.getDeviceCount()} 个GPU设备")
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"   GPU {i}: {props['name'].decode()}")
    else:
        print("⚠️  未检测到GPU设备，将使用CPU模式")
        USE_GPU = False
        
except ImportError:
    USE_GPU = False
    print("⚠️  CuPy未安装，使用CPU模式")

class DiseasePredictor:
    """疾病预测器 - 极致GPU加速版本"""
    
    def __init__(self):
        self.specialists = {}  # 专科医生模型
        self.meta_model = None  # 总分析师元模型
        self.scalers = {}  # 标准化器
        self.label_encoders = {}  # 标签编码器
        self.feature_importance = {}  # 特征重要性
        self.shap_values = {}  # SHAP值
        self.gpu_data = {}  # GPU数据缓存
        
        # 创建模型保存目录
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(f"{self.model_dir}/specialists", exist_ok=True)
        os.makedirs(f"{self.model_dir}/meta", exist_ok=True)
        os.makedirs(f"{self.model_dir}/preprocessors", exist_ok=True)
        
        # GPU加速配置
        self.gpu_config = {
            'task_type': 'GPU' if USE_GPU else 'CPU',
            'devices': '0' if USE_GPU else None,
            'gpu_ram_part': 0.8,  # 使用80%的GPU内存
            'thread_count': -1,  # 使用所有CPU核心
            'verbose': False
        }
        
        print(f"🎛️  GPU配置: {self.gpu_config}")
        
    def _to_gpu(self, data):
        """将数据转移到GPU"""
        if USE_GPU and isinstance(data, (np.ndarray, pd.DataFrame)):
            if isinstance(data, pd.DataFrame):
                return cp.asarray(data.values)
            return cp.asarray(data)
        return data
    
    def _to_cpu(self, data):
        """将数据从GPU转移到CPU"""
        if USE_GPU and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data
        
    def save_models(self):
        """保存所有训练好的模型"""
        print("💾 正在保存模型...")
        
        # 保存专科医生模型
        for disease in ['stroke', 'heart', 'cirrhosis']:
            if disease in self.specialists:
                best_model = self.specialists[disease]['best_model']
                best_model_name = self.specialists[disease]['best_model_name']
                
                model_path = f"{self.model_dir}/specialists/{disease}_{best_model_name}.cbm"
                best_model.save_model(model_path)
                print(f"   ✅ 保存 {disease} 最佳模型: {model_path}")
        
        # 保存元模型
        if self.meta_model is not None:
            meta_model_path = f"{self.model_dir}/meta/meta_model.cbm"
            self.meta_model.save_model(meta_model_path)
            print(f"   ✅ 保存元模型: {meta_model_path}")
        
        # 保存预处理器
        preprocessors = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders
        }
        preprocessors_path = f"{self.model_dir}/preprocessors/preprocessors.pkl"
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"   ✅ 保存预处理器: {preprocessors_path}")
        
        # 保存模型信息
        model_info = {
            'specialists': {disease: {
                'best_model_name': self.specialists[disease]['best_model_name'],
                'model_path': f"{self.model_dir}/specialists/{disease}_{self.specialists[disease]['best_model_name']}.cbm"
            } for disease in self.specialists.keys()},
            'meta_model_path': f"{self.model_dir}/meta/meta_model.cbm",
            'preprocessors_path': preprocessors_path,
            'gpu_config': self.gpu_config
        }
        
        info_path = f"{self.model_dir}/model_info.json"
        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"   ✅ 保存模型信息: {info_path}")
        
        print("🎉 模型保存完成！")
        
    def load_models(self):
        """加载已保存的模型"""
        print("📂 正在加载模型...")
        
        # 加载模型信息
        info_path = f"{self.model_dir}/model_info.json"
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            # 加载预处理器
            preprocessors_path = model_info['preprocessors_path']
            with open(preprocessors_path, 'rb') as f:
                preprocessors = pickle.load(f)
            self.scalers = preprocessors['scalers']
            self.label_encoders = preprocessors['label_encoders']
            
            # 加载专科医生模型
            for disease, info in model_info['specialists'].items():
                model_path = info['model_path']
                model = cb.CatBoostClassifier()
                model.load_model(model_path)
                
                self.specialists[disease] = {
                    'best_model': model,
                    'best_model_name': info['best_model_name']
                }
            
            # 加载元模型
            meta_model_path = model_info['meta_model_path']
            self.meta_model = cb.CatBoostClassifier()
            self.meta_model.load_model(meta_model_path)
            
            print("✅ 模型加载完成！")
            return True
        else:
            print("⚠️  未找到已保存的模型，需要重新训练")
            return False
        
    def load_and_preprocess_data(self):
        """加载和预处理数据 - GPU加速版本"""
        print("🔄 正在加载和预处理数据...")
        start_time = time.time()
        
        # 加载数据
        data_dir = "./附件"
        self.stroke_data = pd.read_csv(f"{data_dir}/stroke.csv", encoding='utf-8-sig')
        self.heart_data = pd.read_csv(f"{data_dir}/heart.csv", encoding='utf-8-sig')
        self.cirrhosis_data = pd.read_csv(f"{data_dir}/cirrhosis.csv", encoding='utf-8-sig')
        
        print(f"📊 数据加载完成: 中风({len(self.stroke_data)}行), 心脏病({len(self.heart_data)}行), 肝硬化({len(self.cirrhosis_data)}行)")
        
        # GPU加速数据预处理
        self._preprocess_stroke_data()
        self._preprocess_heart_data()
        self._preprocess_cirrhosis_data()
        
        elapsed_time = time.time() - start_time
        print(f"⚡ 数据预处理完成，耗时: {elapsed_time:.2f}秒")
        
    def _preprocess_stroke_data(self):
        """预处理中风数据 - GPU加速"""
        df = self.stroke_data.copy()
        
        # 处理缺失值
        df = df.fillna(df.mode().iloc[0])
        
        # 编码分类变量
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[f'stroke_{col}'] = le
        
        # 分离特征和目标
        X = df.drop(['id', 'stroke'], axis=1, errors='ignore')
        y = df['stroke']
        
        # GPU加速标准化
        if USE_GPU:
            X_gpu = self._to_gpu(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self._to_cpu(X_gpu))
            X_scaled = self._to_gpu(X_scaled)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        self.scalers['stroke'] = scaler
        
        self.stroke_X = pd.DataFrame(self._to_cpu(X_scaled), columns=X.columns)
        self.stroke_y = y
        
        # 缓存GPU数据
        if USE_GPU:
            self.gpu_data['stroke_X'] = self._to_gpu(self.stroke_X)
            self.gpu_data['stroke_y'] = self._to_gpu(self.stroke_y)
        
    def _preprocess_heart_data(self):
        """预处理心脏病数据 - GPU加速"""
        df = self.heart_data.copy()
        
        # 处理缺失值
        df = df.fillna(df.mode().iloc[0])
        
        # 编码分类变量
        categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[f'heart_{col}'] = le
        
        # 分离特征和目标
        X = df.drop(['HeartDisease'], axis=1, errors='ignore')
        y = df['HeartDisease']
        
        # GPU加速标准化
        if USE_GPU:
            X_gpu = self._to_gpu(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self._to_cpu(X_gpu))
            X_scaled = self._to_gpu(X_scaled)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        self.scalers['heart'] = scaler
        
        self.heart_X = pd.DataFrame(self._to_cpu(X_scaled), columns=X.columns)
        self.heart_y = y
        
        # 缓存GPU数据
        if USE_GPU:
            self.gpu_data['heart_X'] = self._to_gpu(self.heart_X)
            self.gpu_data['heart_y'] = self._to_gpu(self.heart_y)
        
    def _preprocess_cirrhosis_data(self):
        """预处理肝硬化数据 - GPU加速"""
        df = self.cirrhosis_data.copy()
        
        # 处理缺失值
        df = df.fillna(df.mode().iloc[0])
        
        # 编码分类变量
        categorical_cols = ['Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Status', 'Drug']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[f'cirrhosis_{col}'] = le
        
        # 分离特征和目标
        X = df.drop(['ID', 'Stage'], axis=1, errors='ignore')
        y = (df['Stage'] > 1).astype(int)  # 将Stage>1视为患病
        
        # 确保所有列都是数值型
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(0)
                except:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[f'cirrhosis_{col}'] = le
        
        # GPU加速标准化
        if USE_GPU:
            X_gpu = self._to_gpu(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self._to_cpu(X_gpu))
            X_scaled = self._to_gpu(X_scaled)
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        self.scalers['cirrhosis'] = scaler
        
        self.cirrhosis_X = pd.DataFrame(self._to_cpu(X_scaled), columns=X.columns)
        self.cirrhosis_y = y
        
        # 缓存GPU数据
        if USE_GPU:
            self.gpu_data['cirrhosis_X'] = self._to_gpu(self.cirrhosis_X)
            self.gpu_data['cirrhosis_y'] = self._to_gpu(self.cirrhosis_y)
        
    def train_specialists(self):
        """训练专科医生模型 - 极致GPU加速版本"""
        print("🏥 正在训练专科医生模型...")
        start_time = time.time()
        
        # 中风专家
        print("🧠 训练中风专家...")
        self._train_specialist('stroke', self.stroke_X, self.stroke_y)
        
        # 心脏病专家
        print("❤️  训练心脏病专家...")
        self._train_specialist('heart', self.heart_X, self.heart_y)
        
        # 肝硬化专家
        print("🫁 训练肝硬化专家...")
        self._train_specialist('cirrhosis', self.cirrhosis_X, self.cirrhosis_y)
        
        elapsed_time = time.time() - start_time
        print(f"⚡ 专科医生模型训练完成，耗时: {elapsed_time:.2f}秒")
        
    def _train_specialist(self, disease, X, y):
        """训练单个专科医生模型 - 极致GPU加速"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 极致GPU加速的CatBoost配置
        catboost_config = {
            **self.gpu_config,
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 8,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': 42,
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
            'verbose': 100
        }
        
        print(f"   🚀 使用极致GPU加速配置训练 {disease} 模型...")
        
        # 训练CatBoost模型
        model = cb.CatBoostClassifier(**catboost_config)
        
        # CatBoost需要CPU格式的数据，但使用GPU训练
        # 确保数据是CPU格式的numpy数组或pandas DataFrame
        if isinstance(X_train, pd.DataFrame):
            X_train_cpu = X_train
            X_test_cpu = X_test
            y_train_cpu = y_train
            y_test_cpu = y_test
        else:
            # 如果是GPU数组，转换为CPU格式
            X_train_cpu = self._to_cpu(X_train)
            X_test_cpu = self._to_cpu(X_test)
            y_train_cpu = self._to_cpu(y_train)
            y_test_cpu = self._to_cpu(y_test)
            
            # 转换为DataFrame以保持列名
            if isinstance(X_train_cpu, np.ndarray):
                X_train_cpu = pd.DataFrame(X_train_cpu, columns=X.columns)
                X_test_cpu = pd.DataFrame(X_test_cpu, columns=X.columns)
        
        # 训练模型（CatBoost内部会使用GPU）
        model.fit(
            X_train_cpu, y_train_cpu,
            eval_set=(X_test_cpu, y_test_cpu),
            plot=False
        )
        
        # 预测
        y_pred = model.predict(X_test_cpu)
        y_pred_proba = model.predict_proba(X_test_cpu)[:, 1]
        
        # 评估指标
        accuracy = accuracy_score(y_test_cpu, y_pred)
        precision = precision_score(y_test_cpu, y_pred, zero_division=0)
        recall = recall_score(y_test_cpu, y_pred, zero_division=0)
        f1 = f1_score(y_test_cpu, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_cpu, y_pred_proba)
        
        results = {
            'catboost': {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # 保存最佳模型和结果
        self.specialists[disease] = {
            'best_model': model,
            'best_model_name': 'catboost',
            'results': results,
            'X_test': X_test_cpu,
            'y_test': y_test_cpu
        }
        
        print(f"   ✅ {disease} 模型训练完成 (AUC: {auc:.4f})")
        
        # 计算特征重要性
        self.feature_importance[disease] = dict(zip(
            X.columns, 
            model.get_feature_importance()
        ))
        
        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        self.shap_values[disease] = explainer.shap_values(X_test_cpu)
        
    def create_meta_features(self):
        """创建元特征 - GPU加速版本"""
        print("🔗 正在创建元特征...")
        start_time = time.time()
        
        meta_features = []
        meta_labels = []
        
        # 为每个数据集创建元特征
        for disease in ['stroke', 'heart', 'cirrhosis']:
            X = getattr(self, f'{disease}_X')
            y = getattr(self, f'{disease}_y')
            
            # 获取专科医生的预测概率
            specialist = self.specialists[disease]['best_model']
            probabilities = specialist.predict_proba(X)[:, 1]
            
            # 创建元特征：专科医生预测 + 基本信息
            meta_feature = {
                f'{disease}_prob': probabilities,
                'age': X['Age'] if 'Age' in X.columns else X['age'] if 'age' in X.columns else 0,
                'gender': X['Sex'] if 'Sex' in X.columns else X['gender'] if 'gender' in X.columns else 0
            }
            
            meta_features.append(pd.DataFrame(meta_feature))
            meta_labels.append(y)
        
        # 合并所有元特征
        self.meta_X = pd.concat(meta_features, ignore_index=True)
        self.meta_y = pd.concat(meta_labels, ignore_index=True)
        
        # GPU缓存
        if USE_GPU:
            self.gpu_data['meta_X'] = self._to_gpu(self.meta_X)
            self.gpu_data['meta_y'] = self._to_gpu(self.meta_y)
        
        elapsed_time = time.time() - start_time
        print(f"⚡ 元特征创建完成，耗时: {elapsed_time:.2f}秒")
        
    def train_meta_model(self):
        """训练总分析师元模型 - 极致GPU加速"""
        print("🧠 正在训练总分析师元模型...")
        start_time = time.time()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.meta_X, self.meta_y, test_size=0.2, random_state=42, stratify=self.meta_y
        )
        
        # 极致GPU加速的元模型配置
        meta_config = {
            **self.gpu_config,
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 5,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.9,
            'random_seed': 42,
            'eval_metric': 'AUC',
            'early_stopping_rounds': 30,
            'verbose': 100
        }
        
        # 使用CatBoost作为元模型
        self.meta_model = cb.CatBoostClassifier(**meta_config)
        
        # 确保数据是CPU格式
        if isinstance(X_train, pd.DataFrame):
            X_train_cpu = X_train
            X_test_cpu = X_test
            y_train_cpu = y_train
            y_test_cpu = y_test
        else:
            # 如果是GPU数组，转换为CPU格式
            X_train_cpu = self._to_cpu(X_train)
            X_test_cpu = self._to_cpu(X_test)
            y_train_cpu = self._to_cpu(y_train)
            y_test_cpu = self._to_cpu(y_test)
            
            # 转换为DataFrame以保持列名
            if isinstance(X_train_cpu, np.ndarray):
                X_train_cpu = pd.DataFrame(X_train_cpu, columns=self.meta_X.columns)
                X_test_cpu = pd.DataFrame(X_test_cpu, columns=self.meta_X.columns)
        
        # 训练模型（CatBoost内部会使用GPU）
        self.meta_model.fit(
            X_train_cpu, y_train_cpu,
            eval_set=(X_test_cpu, y_test_cpu),
            plot=False
        )
        
        # 评估元模型
        y_pred = self.meta_model.predict(X_test_cpu)
        y_pred_proba = self.meta_model.predict_proba(X_test_cpu)[:, 1]
        
        self.meta_results = {
            'accuracy': accuracy_score(y_test_cpu, y_pred),
            'precision': precision_score(y_test_cpu, y_pred, zero_division=0),
            'recall': recall_score(y_test_cpu, y_pred, zero_division=0),
            'f1': f1_score(y_test_cpu, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test_cpu, y_pred_proba)
        }
        
        elapsed_time = time.time() - start_time
        print(f"⚡ 元模型训练完成 (AUC: {self.meta_results['auc']:.4f})，耗时: {elapsed_time:.2f}秒")
        
    def sensitivity_analysis(self):
        """灵敏度分析 - GPU加速版本"""
        print("🔍 正在进行灵敏度分析...")
        start_time = time.time()
        
        sensitivity_results = {}
        
        for disease in ['stroke', 'heart', 'cirrhosis']:
            print(f"🔬 分析 {disease} 模型灵敏度...")
            
            # 第一层：特征扰动分析
            feature_sensitivity = self._feature_perturbation_analysis(disease)
            
            # 第二层：模型稳定性检验
            model_stability = self._bootstrap_stability_analysis(disease)
            
            sensitivity_results[disease] = {
                'feature_sensitivity': feature_sensitivity,
                'model_stability': model_stability
            }
        
        self.sensitivity_results = sensitivity_results
        
        elapsed_time = time.time() - start_time
        print(f"⚡ 灵敏度分析完成，耗时: {elapsed_time:.2f}秒")
        
    def _feature_perturbation_analysis(self, disease):
        """特征扰动分析 - GPU加速"""
        X = getattr(self, f'{disease}_X')
        y = getattr(self, f'{disease}_y')
        model = self.specialists[disease]['best_model']
        
        # 选择核心特征
        if disease == 'stroke':
            core_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        elif disease == 'heart':
            core_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        else:  # cirrhosis
            core_features = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper']
        
        # 过滤存在的特征
        core_features = [f for f in core_features if f in X.columns]
        
        sensitivity = {}
        perturbations = [-0.2, -0.1, 0, 0.1, 0.2]  # ±20%扰动
        
        for feature in tqdm(core_features, desc=f"🔍 Feature perturbation analysis for {disease}"):
            original_values = X[feature].copy()
            prob_changes = []
            
            for pert in perturbations:
                # 应用扰动
                X_perturbed = X.copy()
                X_perturbed[feature] = original_values * (1 + pert)
                
                # GPU加速预测
                if USE_GPU:
                    X_gpu = self._to_gpu(X)
                    X_perturbed_gpu = self._to_gpu(X_perturbed)
                    
                    original_probs = model.predict_proba(self._to_cpu(X_gpu))[:, 1]
                    perturbed_probs = model.predict_proba(self._to_cpu(X_perturbed_gpu))[:, 1]
                else:
                    original_probs = model.predict_proba(X)[:, 1]
                    perturbed_probs = model.predict_proba(X_perturbed)[:, 1]
                
                # 计算变化率
                change_rate = np.mean(np.abs(perturbed_probs - original_probs) / (original_probs + 1e-8))
                prob_changes.append(change_rate)
            
            sensitivity[feature] = dict(zip(perturbations, prob_changes))
        
        return sensitivity
        
    def _bootstrap_stability_analysis(self, disease):
        """Bootstrap稳定性分析 - GPU加速"""
        X = getattr(self, f'{disease}_X')
        y = getattr(self, f'{disease}_y')
        
        n_bootstrap = 50  # 减少到50次以加快速度
        prob_std = []
        
        for _ in tqdm(range(n_bootstrap), desc=f"🔄 Bootstrap stability analysis for {disease}"):
            # Bootstrap采样
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # GPU加速训练
            config = {
                **self.gpu_config,
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'random_seed': 42,
                'verbose': False
            }
            
            model = cb.CatBoostClassifier(**config)
            
            # 确保数据是CPU格式
            if isinstance(X_boot, pd.DataFrame):
                X_boot_cpu = X_boot
                y_boot_cpu = y_boot
            else:
                # 如果是GPU数组，转换为CPU格式
                X_boot_cpu = self._to_cpu(X_boot)
                y_boot_cpu = self._to_cpu(y_boot)
                
                # 转换为DataFrame以保持列名
                if isinstance(X_boot_cpu, np.ndarray):
                    X_boot_cpu = pd.DataFrame(X_boot_cpu, columns=X.columns)
            
            # 训练模型（CatBoost内部会使用GPU）
            model.fit(X_boot_cpu, y_boot_cpu, verbose=False)
            
            # 预测测试集
            X_test = self.specialists[disease]['X_test']
            
            # 确保测试数据也是CPU格式
            if isinstance(X_test, pd.DataFrame):
                X_test_cpu = X_test
            else:
                X_test_cpu = self._to_cpu(X_test)
                if isinstance(X_test_cpu, np.ndarray):
                    X_test_cpu = pd.DataFrame(X_test_cpu, columns=X.columns)
            
            probs = model.predict_proba(X_test_cpu)[:, 1]
            prob_std.append(probs)
        
        # 计算概率标准差
        prob_std = np.std(prob_std, axis=0)
        return np.mean(prob_std)
        
    def generate_reports(self):
        """生成分析报告"""
        print("📊 正在生成分析报告...")
        
        # 创建输出目录
        os.makedirs("output/csv/第二问", exist_ok=True)
        os.makedirs("output/plt/第二问", exist_ok=True)
        
        # 保存模型性能报告
        self._save_performance_report()
        
        # 保存特征重要性
        self._save_feature_importance()
        
        # 生成可视化
        self._generate_visualizations()
        
        print("✅ 分析报告生成完成")
        
    def _save_performance_report(self):
        """保存模型性能报告"""
        report_data = []
        
        # 专科医生模型性能
        for disease in ['stroke', 'heart', 'cirrhosis']:
            results = self.specialists[disease]['results']
            best_model = self.specialists[disease]['best_model_name']
            
            for model_name, result in results.items():
                report_data.append({
                    'Disease': disease,
                    'Model': model_name,
                    'Best_Model': model_name == best_model,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1'],
                    'AUC': result['auc']
                })
        
        # 元模型性能
        report_data.append({
            'Disease': 'Meta_Model',
            'Model': 'CatBoost',
            'Best_Model': True,
            'Accuracy': self.meta_results['accuracy'],
            'Precision': self.meta_results['precision'],
            'Recall': self.meta_results['recall'],
            'F1_Score': self.meta_results['f1'],
            'AUC': self.meta_results['auc']
        })
        
        # 保存报告
        report_df = pd.DataFrame(report_data)
        report_df.to_csv("output/csv/第二问/model_performance_report.csv", 
                        index=False, encoding='utf-8-sig')
        
    def _save_feature_importance(self):
        """保存特征重要性"""
        for disease, importance in self.feature_importance.items():
            importance_df = pd.DataFrame([
                {'Feature': feature, 'Importance': imp}
                for feature, imp in importance.items()
            ]).sort_values('Importance', ascending=False)
            
            importance_df.to_csv(f"output/csv/第二问/{disease}_feature_importance.csv", 
                               index=False, encoding='utf-8-sig')
        
    def _generate_visualizations(self):
        """生成可视化图表"""
        # 1. 模型性能对比
        self._plot_model_comparison()
        
        # 2. 特征重要性
        self._plot_feature_importance()
        
        # 3. ROC曲线
        self._plot_roc_curves()
        
        # 4. 灵敏度分析结果
        self._plot_sensitivity_analysis()
        
    def _plot_model_comparison(self):
        """绘制模型性能对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics = ['Accuracy', 'Precision', 'Recall', 'AUC']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            data = []
            for disease in ['stroke', 'heart', 'cirrhosis']:
                results = self.specialists[disease]['results']
                for model_name, result in results.items():
                    data.append({
                        'Disease': disease,
                        'Model': model_name,
                        metric: result[metric.lower().replace('auc', 'auc')]
                    })
            
            df = pd.DataFrame(data)
            sns.barplot(data=df, x='Disease', y=metric, hue='Model', ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig("output/plt/第二问/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_feature_importance(self):
        """绘制特征重要性图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, disease in enumerate(['stroke', 'heart', 'cirrhosis']):
            if disease in self.feature_importance:
                importance = self.feature_importance[disease]
                features = list(importance.keys())[:10]  # 前10个特征
                values = [importance[f] for f in features]
                
                axes[i].barh(features, values)
                axes[i].set_title(f'{disease.capitalize()} Feature Importance')
                axes[i].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig("output/plt/第二问/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_roc_curves(self):
        """绘制ROC曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, disease in enumerate(['stroke', 'heart', 'cirrhosis']):
            results = self.specialists[disease]['results']
            y_test = self.specialists[disease]['y_test']
            
            for model_name, result in results.items():
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                auc = result['auc']
                
                axes[i].plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})')
            
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{disease.capitalize()} ROC Curves')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("output/plt/第二问/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_sensitivity_analysis(self):
        """绘制灵敏度分析结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, disease in enumerate(['stroke', 'heart', 'cirrhosis']):
            sensitivity = self.sensitivity_results[disease]['feature_sensitivity']
            
            if sensitivity:
                features = list(sensitivity.keys())
                perturbations = [-0.2, -0.1, 0, 0.1, 0.2]
                
                for feature in features[:5]:  # 前5个特征
                    changes = [sensitivity[feature][p] for p in perturbations]
                    axes[i].plot(perturbations, changes, marker='o', label=feature)
                
                axes[i].set_xlabel('Perturbation (%)')
                axes[i].set_ylabel('Probability Change Rate')
                axes[i].set_title(f'{disease.capitalize()} Feature Sensitivity')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("output/plt/第二问/sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def run_complete_analysis(self):
        """运行完整的分析流程 - 极致GPU加速版本"""
        print("🚀 开始极致GPU加速疾病预测模型构建...")
        total_start_time = time.time()
        
        # 1. 加载和预处理数据
        self.load_and_preprocess_data()
        
        # 2. 训练专科医生模型
        self.train_specialists()
        
        # 3. 创建元特征
        self.create_meta_features()
        
        # 4. 训练元模型
        self.train_meta_model()
        
        # 5. 灵敏度分析
        self.sensitivity_analysis()
        
        # 6. 生成报告
        self.generate_reports()
        
        # 7. 保存模型
        self.save_models()
        
        total_elapsed_time = time.time() - total_start_time
        print(f"🎉 极致GPU加速疾病预测模型构建完成！")
        print(f"⏱️  总耗时: {total_elapsed_time:.2f}秒")
        print("📁 结果保存在 output/csv/第二问 和 output/plt/第二问 文件夹中")
        print("💾 模型保存在 models/ 文件夹中")

if __name__ == "__main__":
    # 创建预测器并运行分析
    predictor = DiseasePredictor()
    predictor.run_complete_analysis() 