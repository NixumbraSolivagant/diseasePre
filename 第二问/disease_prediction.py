# -*- coding: utf-8 -*-
"""
第二问：不同疾病预测模型的构建
分而治之的集成学习方案：
1. 专科医生模型（单疾病专家）
2. 总分析师元模型（结果融合）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
import joblib
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 设置GPU加速（如果可用）
try:
    import cupy as cp
    USE_GPU = True
    print("GPU加速已启用")
except ImportError:
    USE_GPU = False
    print("使用CPU模式")

class DiseasePredictor:
    """疾病预测器 - 实现分而治之的集成学习方案"""
    
    def __init__(self):
        self.specialists = {}  # 专科医生模型
        self.meta_model = None  # 总分析师元模型
        self.scalers = {}  # 标准化器
        self.label_encoders = {}  # 标签编码器
        self.feature_importance = {}  # 特征重要性
        self.shap_values = {}  # SHAP值
        
        # 创建模型保存目录
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(f"{self.model_dir}/specialists", exist_ok=True)
        os.makedirs(f"{self.model_dir}/meta", exist_ok=True)
        os.makedirs(f"{self.model_dir}/preprocessors", exist_ok=True)
        
    def save_models(self):
        """保存所有训练好的模型"""
        print("正在保存模型...")
        
        # 保存专科医生模型
        for disease in ['stroke', 'heart', 'cirrhosis']:
            if disease in self.specialists:
                # 保存最佳模型
                best_model = self.specialists[disease]['best_model']
                best_model_name = self.specialists[disease]['best_model_name']
                
                # 根据模型类型选择保存方法
                if best_model_name in ['xgboost', 'lightgbm']:
                    model_path = f"{self.model_dir}/specialists/{disease}_{best_model_name}.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(best_model, f)
                else:
                    model_path = f"{self.model_dir}/specialists/{disease}_{best_model_name}.joblib"
                    joblib.dump(best_model, model_path)
                
                print(f"  保存 {disease} 最佳模型: {model_path}")
        
        # 保存元模型
        if self.meta_model is not None:
            meta_model_path = f"{self.model_dir}/meta/meta_model.pkl"
            with open(meta_model_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
            print(f"  保存元模型: {meta_model_path}")
        
        # 保存预处理器
        preprocessors = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders
        }
        preprocessors_path = f"{self.model_dir}/preprocessors/preprocessors.pkl"
        with open(preprocessors_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        print(f"  保存预处理器: {preprocessors_path}")
        
        # 保存模型信息
        model_info = {
            'specialists': {disease: {
                'best_model_name': self.specialists[disease]['best_model_name'],
                'model_path': f"{self.model_dir}/specialists/{disease}_{self.specialists[disease]['best_model_name']}.pkl"
            } for disease in self.specialists.keys()},
            'meta_model_path': f"{self.model_dir}/meta/meta_model.pkl",
            'preprocessors_path': preprocessors_path
        }
        
        info_path = f"{self.model_dir}/model_info.json"
        import json
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        print(f"  保存模型信息: {info_path}")
        
        print("模型保存完成！")
        
    def load_models(self):
        """加载已保存的模型"""
        print("正在加载模型...")
        
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
                if model_path.endswith('.pkl'):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                else:
                    model = joblib.load(model_path)
                
                self.specialists[disease] = {
                    'best_model': model,
                    'best_model_name': info['best_model_name']
                }
            
            # 加载元模型
            meta_model_path = model_info['meta_model_path']
            with open(meta_model_path, 'rb') as f:
                self.meta_model = pickle.load(f)
            
            print("模型加载完成！")
            return True
        else:
            print("未找到已保存的模型，需要重新训练")
            return False
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("正在加载和预处理数据...")
        
        # 加载数据
        data_dir = "./附件"
        self.stroke_data = pd.read_csv(f"{data_dir}/stroke.csv", encoding='utf-8-sig')
        self.heart_data = pd.read_csv(f"{data_dir}/heart.csv", encoding='utf-8-sig')
        self.cirrhosis_data = pd.read_csv(f"{data_dir}/cirrhosis.csv", encoding='utf-8-sig')
        
        # 数据预处理
        self._preprocess_stroke_data()
        self._preprocess_heart_data()
        self._preprocess_cirrhosis_data()
        
        print("数据预处理完成")
        
    def _preprocess_stroke_data(self):
        """预处理中风数据"""
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
        
        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['stroke'] = scaler
        
        self.stroke_X = pd.DataFrame(X_scaled, columns=X.columns)
        self.stroke_y = y
        
    def _preprocess_heart_data(self):
        """预处理心脏病数据"""
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
        
        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['heart'] = scaler
        
        self.heart_X = pd.DataFrame(X_scaled, columns=X.columns)
        self.heart_y = y
        
    def _preprocess_cirrhosis_data(self):
        """预处理肝硬化数据"""
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
                # 如果还有字符串列，尝试转换为数值
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    # 如果转换后有NaN，用0填充
                    X[col] = X[col].fillna(0)
                except:
                    # 如果无法转换，用LabelEncoder编码
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[f'cirrhosis_{col}'] = le
        
        # 标准化数值特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['cirrhosis'] = scaler
        
        self.cirrhosis_X = pd.DataFrame(X_scaled, columns=X.columns)
        self.cirrhosis_y = y
        
    def train_specialists(self):
        """训练专科医生模型（第一步：分而治之）"""
        print("正在训练专科医生模型...")
        
        # 中风专家
        print("训练中风专家...")
        self._train_specialist('stroke', self.stroke_X, self.stroke_y)
        
        # 心脏病专家
        print("训练心脏病专家...")
        self._train_specialist('heart', self.heart_X, self.heart_y)
        
        # 肝硬化专家
        print("训练肝硬化专家...")
        self._train_specialist('cirrhosis', self.cirrhosis_X, self.cirrhosis_y)
        
        print("专科医生模型训练完成")
        
    def _train_specialist(self, disease, X, y):
        """训练单个专科医生模型"""
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 定义模型
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        best_model = None
        best_score = 0
        results = {}
        
        # 训练和评估所有模型
        for name, model in tqdm(models.items(), desc=f"Training {disease} models"):
            print(f"  训练 {name} 模型...")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # 评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_pred_proba': y_pred_proba
            }
            
            # 选择最佳模型（基于AUC）
            if auc > best_score:
                best_score = auc
                best_model = name
        
        # 保存最佳模型和结果
        self.specialists[disease] = {
            'best_model': results[best_model]['model'],
            'best_model_name': best_model,
            'results': results,
            'X_test': X_test,
            'y_test': y_test
        }
        
        print(f"  {disease} 最佳模型: {best_model} (AUC: {best_score:.4f})")
        
        # 计算特征重要性（如果模型支持）
        if hasattr(results[best_model]['model'], 'feature_importances_'):
            self.feature_importance[disease] = dict(zip(
                X.columns, 
                results[best_model]['model'].feature_importances_
            ))
        
        # 计算SHAP值
        if best_model in ['xgboost', 'lightgbm', 'random_forest']:
            explainer = shap.TreeExplainer(results[best_model]['model'])
            self.shap_values[disease] = explainer.shap_values(X_test)
        
    def create_meta_features(self):
        """创建元特征（第二步：结果融合的准备）"""
        print("正在创建元特征...")
        
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
        
        print("元特征创建完成")
        
    def train_meta_model(self):
        """训练总分析师元模型（第二步：结果融合）"""
        print("正在训练总分析师元模型...")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            self.meta_X, self.meta_y, test_size=0.2, random_state=42, stratify=self.meta_y
        )
        
        # 使用XGBoost作为元模型
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # 训练元模型
        self.meta_model.fit(X_train, y_train)
        
        # 评估元模型
        y_pred = self.meta_model.predict(X_test)
        y_pred_proba = self.meta_model.predict_proba(X_test)[:, 1]
        
        self.meta_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"元模型训练完成 (AUC: {self.meta_results['auc']:.4f})")
        
    def sensitivity_analysis(self):
        """灵敏度分析"""
        print("正在进行灵敏度分析...")
        
        sensitivity_results = {}
        
        for disease in ['stroke', 'heart', 'cirrhosis']:
            print(f"分析 {disease} 模型灵敏度...")
            
            # 第一层：特征扰动分析
            feature_sensitivity = self._feature_perturbation_analysis(disease)
            
            # 第二层：模型稳定性检验
            model_stability = self._bootstrap_stability_analysis(disease)
            
            sensitivity_results[disease] = {
                'feature_sensitivity': feature_sensitivity,
                'model_stability': model_stability
            }
        
        self.sensitivity_results = sensitivity_results
        print("灵敏度分析完成")
        
    def _feature_perturbation_analysis(self, disease):
        """特征扰动分析"""
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
        
        for feature in tqdm(core_features, desc=f"Feature perturbation analysis for {disease}"):
            original_values = X[feature].copy()
            prob_changes = []
            
            for pert in perturbations:
                # 应用扰动
                X_perturbed = X.copy()
                X_perturbed[feature] = original_values * (1 + pert)
                
                # 预测
                original_probs = model.predict_proba(X)[:, 1]
                perturbed_probs = model.predict_proba(X_perturbed)[:, 1]
                
                # 计算变化率
                change_rate = np.mean(np.abs(perturbed_probs - original_probs) / (original_probs + 1e-8))
                prob_changes.append(change_rate)
            
            sensitivity[feature] = dict(zip(perturbations, prob_changes))
        
        return sensitivity
        
    def _bootstrap_stability_analysis(self, disease):
        """Bootstrap稳定性分析"""
        X = getattr(self, f'{disease}_X')
        y = getattr(self, f'{disease}_y')
        
        n_bootstrap = 100  # 减少到100次以加快速度
        prob_std = []
        
        for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap stability analysis for {disease}"):
            # Bootstrap采样
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # 训练模型
            model = xgb.XGBClassifier(n_estimators=50, random_state=42, verbose=-1)
            model.fit(X_boot, y_boot)
            
            # 预测测试集
            X_test = self.specialists[disease]['X_test']
            probs = model.predict_proba(X_test)[:, 1]
            prob_std.append(probs)
        
        # 计算概率标准差
        prob_std = np.std(prob_std, axis=0)
        return np.mean(prob_std)
        
    def generate_reports(self):
        """生成分析报告"""
        print("正在生成分析报告...")
        
        # 创建输出目录
        os.makedirs("output/csv/第二问", exist_ok=True)
        os.makedirs("output/plt/第二问", exist_ok=True)
        
        # 保存模型性能报告
        self._save_performance_report()
        
        # 保存特征重要性
        self._save_feature_importance()
        
        # 生成可视化
        self._generate_visualizations()
        
        print("分析报告生成完成")
        
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
            'Model': 'XGBoost',
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
        """运行完整的分析流程"""
        print("开始疾病预测模型构建...")
        
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
        
        print("疾病预测模型构建完成！")
        print("结果保存在 output/csv/第二问 和 output/plt/第二问 文件夹中")

if __name__ == "__main__":
    # 创建预测器并运行分析
    predictor = DiseasePredictor()
    predictor.run_complete_analysis() 