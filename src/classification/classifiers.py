"""
分类模块
负责训练和评估不同的机器学习分类器
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import warnings
from loguru import logger

from ..config.config_manager import ExperimentConfig

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化特征提取器

        Args:
            config: 实验配置
        """
        self.config = config
        self.feature_config = config.feature_extraction
        self.tfidf_vectorizer = None
        self.embedding_model = None
        self.tokenizer = None

    def fit_tfidf(self, texts: List[str]) -> 'FeatureExtractor':
        """
        训练TF-IDF向量化器

        Args:
            texts: 文本列表

        Returns:
            FeatureExtractor: 自身
        """
        tfidf_config = self.feature_config.get('tfidf', {})

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_config.get('max_features', 10000),
            ngram_range=tuple(tfidf_config.get('ngram_range', [1, 2])),
            stop_words=tfidf_config.get('stop_words', 'english'),
            lowercase=tfidf_config.get('lowercase', True),
            min_df=2,
            max_df=0.95
        )

        logger.info("训练TF-IDF向量化器")
        self.tfidf_vectorizer.fit(texts)
        logger.info(f"TF-IDF特征维度: {len(self.tfidf_vectorizer.vocabulary_)}")

        return self

    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """
        使用TF-IDF转换文本

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: TF-IDF特征矩阵
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF向量化器未训练")

        return self.tfidf_vectorizer.transform(texts).toarray()

    def load_embedding_model(self):
        """加载embedding模型"""
        embedding_config = self.feature_config.get('embeddings', {})
        model_name = embedding_config.get('model_name', 'all-MiniLM-L6-v2')

        logger.info(f"加载embedding模型: {model_name}")

        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding模型加载成功")
        except ImportError:
            logger.warning("sentence-transformers未安装，使用transformers库")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)

    def transform_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        使用embedding转换文本

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: embedding特征矩阵
        """
        if self.embedding_model is None:
            self.load_embedding_model()

        logger.info(f"生成embeddings: {len(texts)} 个文本")

        if hasattr(self.embedding_model, 'encode'):
            # sentence-transformers
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        else:
            # transformers
            embeddings = []
            embedding_config = self.feature_config.get('embeddings', {})
            batch_size = embedding_config.get('batch_size', 32)

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=embedding_config.get('max_seq_length', 512),
                    return_tensors='pt'
                )

                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    # 使用CLS token的embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.extend(batch_embeddings)

            embeddings = np.array(embeddings)

        logger.info(f"Embeddings维度: {embeddings.shape}")
        return embeddings

    def extract_features(self, data: pd.DataFrame, feature_type: str = 'tfidf') -> np.ndarray:
        """
        提取特征

        Args:
            data: 数据
            feature_type: 特征类型 ('tfidf', 'embeddings')

        Returns:
            np.ndarray: 特征矩阵
        """
        # 合并subject和body
        texts = (data['subject'].fillna('') + ' ' + data['body'].fillna('')).tolist()

        if feature_type == 'tfidf':
            return self.transform_tfidf(texts)
        elif feature_type == 'embeddings':
            return self.transform_embeddings(texts)
        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")


class BaseClassifier:
    """基础分类器类"""

    def __init__(self, config: ExperimentConfig, classifier_name: str):
        """
        初始化分类器

        Args:
            config: 实验配置
            classifier_name: 分类器名称
        """
        self.config = config
        self.classifier_name = classifier_name
        self.classifier_config = config.classifiers[classifier_name]
        self.model = None
        self.feature_extractor = FeatureExtractor(config)
        self.scaler = StandardScaler()

    def _prepare_features(self, data: pd.DataFrame, fit_extractor: bool = False) -> np.ndarray:
        """
        准备特征

        Args:
            data: 数据
            fit_extractor: 是否训练特征提取器

        Returns:
            np.ndarray: 特征矩阵
        """
        if fit_extractor:
            texts = (data['subject'].fillna('') + ' ' + data['body'].fillna('')).tolist()
            self.feature_extractor.fit_tfidf(texts)

        features = self.feature_extractor.extract_features(data, 'tfidf')
        return features

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BaseClassifier':
        """
        训练分类器

        Args:
            X_train: 训练特征
            y_train: 训练标签

        Returns:
            BaseClassifier: 自身
        """
        raise NotImplementedError

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            X_test: 测试特征

        Returns:
            np.ndarray: 预测结果
        """
        raise NotImplementedError

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        预测概率

        Args:
            X_test: 测试特征

        Returns:
            np.ndarray: 预测概率
        """
        raise NotImplementedError

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        评估分类器

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            Dict[str, float]: 评估指标
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        # 计算各种指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
        }

        # ROC-AUC (需要概率)
        if y_pred_proba is not None and y_pred_proba.shape[1] > 1:
            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            metrics['auc_pr'] = average_precision_score(y_test, y_pred_proba[:, 1])

        # 平衡准确率
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['balanced_accuracy'] = (sensitivity + specificity) / 2

        return metrics


class SVMClassifier(BaseClassifier):
    """SVM分类器"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, 'svm')

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'SVMClassifier':
        """训练SVM分类器"""
        logger.info("训练SVM分类器")

        # 准备特征
        X_features = self._prepare_features(X_train, fit_extractor=True)

        # 标准化特征
        X_features_scaled = self.scaler.fit_transform(X_features)

        # 创建SVM模型
        svm_params = self.classifier_config.get('params', {})
        self.model = SVC(
            C=svm_params.get('C', 1.0),
            kernel=svm_params.get('kernel', 'rbf'),
            class_weight=svm_params.get('class_weight', 'balanced'),
            random_state=svm_params.get('random_state', 42),
            probability=True  # 启用概率预测
        )

        # 训练模型
        self.model.fit(X_features_scaled, y_train)

        logger.info("SVM训练完成")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """SVM预测"""
        X_features = self._prepare_features(X_test, fit_extractor=False)
        X_features_scaled = self.scaler.transform(X_features)
        return self.model.predict(X_features_scaled)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """SVM概率预测"""
        X_features = self._prepare_features(X_test, fit_extractor=False)
        X_features_scaled = self.scaler.transform(X_features)
        return self.model.predict_proba(X_features_scaled)


class RandomForestClassifier_Custom(BaseClassifier):
    """随机森林分类器"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, 'random_forest')

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'RandomForestClassifier_Custom':
        """训练随机森林分类器"""
        logger.info("训练随机森林分类器")

        # 准备特征
        X_features = self._prepare_features(X_train, fit_extractor=True)

        # 创建随机森林模型
        rf_params = self.classifier_config.get('params', {})
        self.model = RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 100),
            max_depth=rf_params.get('max_depth', 10),
            min_samples_split=rf_params.get('min_samples_split', 5),
            min_samples_leaf=rf_params.get('min_samples_leaf', 2),
            class_weight=rf_params.get('class_weight', 'balanced'),
            random_state=rf_params.get('random_state', 42),
            n_jobs=-1
        )

        # 训练模型
        self.model.fit(X_features, y_train)

        logger.info("随机森林训练完成")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """随机森林预测"""
        X_features = self._prepare_features(X_test, fit_extractor=False)
        return self.model.predict(X_features)

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """随机森林概率预测"""
        X_features = self._prepare_features(X_test, fit_extractor=False)
        return self.model.predict_proba(X_features)


class DeepLearningClassifier(BaseClassifier):
    """深度学习分类器"""

    def __init__(self, config: ExperimentConfig):
        super().__init__(config, 'deep_learning')
        # 优化设备选择：CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Deep Learning分类器使用设备: {self.device}")

    def _create_model(self, input_dim: int) -> nn.Module:
        """创建神经网络模型"""
        dl_params = self.classifier_config.get('params', {})
        hidden_layers = dl_params.get('hidden_layers', [128, 64, 32])
        dropout_rate = dl_params.get('dropout_rate', 0.3)

        layers = []
        prev_dim = input_dim

        # 隐藏层
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 2))

        return nn.Sequential(*layers)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'DeepLearningClassifier':
        """训练深度学习分类器"""
        logger.info("训练深度学习分类器")

        # 准备特征
        X_features = self._prepare_features(X_train, fit_extractor=True)
        X_features_scaled = self.scaler.fit_transform(X_features)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_features_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_train.values).to(self.device)

        # 创建数据加载器
        dl_params = self.classifier_config.get('params', {})
        batch_size = dl_params.get('batch_size', 32)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 创建模型
        self.model = self._create_model(X_features.shape[1]).to(self.device)

        # 设置优化器和损失函数
        learning_rate = dl_params.get('learning_rate', 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 类别权重处理
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (2 * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        # 训练参数
        epochs = dl_params.get('epochs', 50)
        patience = dl_params.get('early_stopping_patience', 10)

        # 训练循环
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        logger.info("深度学习训练完成")
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """深度学习预测"""
        X_features = self._prepare_features(X_test, fit_extractor=False)
        X_features_scaled = self.scaler.transform(X_features)
        X_tensor = torch.FloatTensor(X_features_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """深度学习概率预测"""
        X_features = self._prepare_features(X_test, fit_extractor=False)
        X_features_scaled = self.scaler.transform(X_features)
        X_tensor = torch.FloatTensor(X_features_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()


class ClassificationExperiment:
    """分类实验管理器"""

    def __init__(self, config: ExperimentConfig):
        """
        初始化实验管理器

        Args:
            config: 实验配置
        """
        self.config = config
        self.classifiers = self._initialize_classifiers()

    def _initialize_classifiers(self) -> Dict[str, BaseClassifier]:
        """初始化所有分类器"""
        classifiers = {}

        for name, classifier_config in self.config.classifiers.items():
            if not classifier_config.get('enabled', True):
                continue

            if name == 'svm':
                classifiers[name] = SVMClassifier(self.config)
            elif name == 'random_forest':
                classifiers[name] = RandomForestClassifier_Custom(self.config)
            elif name == 'deep_learning':
                classifiers[name] = DeepLearningClassifier(self.config)
            else:
                logger.warning(f"未知的分类器: {name}")

        logger.info(f"初始化分类器: {list(classifiers.keys())}")
        return classifiers

    def run_single_experiment(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        运行单个分类实验

        Args:
            train_data: 训练数据
            test_data: 测试数据
            experiment_id: 实验ID

        Returns:
            Dict[str, Any]: 实验结果
        """
        logger.info(f"运行分类实验: {experiment_id}")

        results = {
            'experiment_id': experiment_id,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_spam_ratio': (train_data['label'] == 1).mean(),
            'test_spam_ratio': (test_data['label'] == 1).mean(),
            'classifiers': {}
        }

        # 准备数据
        X_train = train_data[['subject', 'body']].copy()
        y_train = train_data['label']
        X_test = test_data[['subject', 'body']].copy()
        y_test = test_data['label']

        # 运行每个分类器
        for classifier_name, classifier in self.classifiers.items():
            logger.info(f"运行分类器: {classifier_name}")

            try:
                # 训练分类器
                classifier.fit(X_train, y_train)

                # 评估分类器
                metrics = classifier.evaluate(X_test, y_test)

                # 记录结果
                results['classifiers'][classifier_name] = {
                    'metrics': metrics,
                    'success': True,
                    'error': None
                }

                logger.info(f"{classifier_name} F1-score: {metrics['f1_score']:.4f}")

            except Exception as e:
                logger.error(f"分类器 {classifier_name} 运行失败: {e}")
                results['classifiers'][classifier_name] = {
                    'metrics': None,
                    'success': False,
                    'error': str(e)
                }

        return results

    def run_batch_experiments(
        self,
        experiment_configs: List[Dict[str, Any]],
        datasets_dir: str,
        output_dir: str,
        group_id: int = None  # 添加group_id参数用于独立保存
    ) -> Dict[str, Any]:
        """
        批量运行实验（每个group独立保存，避免并发冲突）

        Args:
            experiment_configs: 实验配置列表
            datasets_dir: 数据集目录
            output_dir: 输出目录
            group_id: 当前group ID（用于生成独立文件名）

        Returns:
            Dict[str, Any]: 批量实验结果
        """
        logger.info(f"批量运行实验: {len(experiment_configs)} 个配置")

        all_results = []
        summary = {
            'total_experiments': len(experiment_configs),
            'successful_experiments': 0,
            'failed_experiments': 0,
            'results_file': None
        }

        # 准备结果文件路径（独立文件，按group分开）
        os.makedirs(output_dir, exist_ok=True)

        # 如果指定了group_id，使用独立文件名
        if group_id is not None and len(experiment_configs) > 0:
            strategy = experiment_configs[0]['strategy']
            prompt = experiment_configs[0]['prompt']
            results_file = os.path.join(
                output_dir,
                f"{self.config.name}_{prompt}_{strategy}_group{group_id}_results.json"
            )
        else:
            results_file = os.path.join(output_dir, f"{self.config.name}_classification_results.json")

        summary['results_file'] = results_file

        from ..data_processing.dataset_builder import DatasetBuilder
        builder = DatasetBuilder(self.config)

        for i, exp_config in enumerate(experiment_configs):
            try:
                # 加载数据集
                train_data, test_data = builder.load_dataset_for_experiment(
                    datasets_dir,
                    exp_config['strategy'],
                    exp_config['prompt'],
                    exp_config['synthetic_ratio'],
                    exp_config['group_id'],
                    exp_config['trial']
                )

                # 创建实验ID
                experiment_id = (
                    f"s{exp_config['strategy']}_"
                    f"r{exp_config['synthetic_ratio']}_"
                    f"g{exp_config['group_id']}_"
                    f"t{exp_config['trial']}"
                )

                # 运行实验
                result = self.run_single_experiment(train_data, test_data, experiment_id)
                result.update(exp_config)  # 添加配置信息

                all_results.append(result)
                summary['successful_experiments'] += 1

                logger.info(f"实验 {i+1}/{len(experiment_configs)} 完成")

            except Exception as e:
                logger.error(f"实验 {i+1}/{len(experiment_configs)} 失败: {e}")
                summary['failed_experiments'] += 1

        # 最终保存（每个group独立保存，无需锁）
        batch_results = {
            'experiment_name': self.config.name,
            'summary': summary,
            'results': all_results,
            'config': {
                'classifiers': list(self.classifiers.keys()),
                'evaluation_metrics': self.config.evaluation.get('metrics', [])
            }
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"批量实验完成: 成功{summary['successful_experiments']}, 失败{summary['failed_experiments']}")
        logger.info(f"结果已保存到: {results_file}")

        return batch_results

    def _save_checkpoint(self, results_file: str, new_results: List[Dict[str, Any]]):
        """保存checkpoint到结果文件（追加模式）"""
        existing_results = []

        # 读取已有结果
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_results = data.get('results', [])
            except:
                pass

        # 合并结果（去重）
        existing_keys = {
            (r['strategy'], r['synthetic_ratio'], r['group_id'], r['trial'])
            for r in existing_results
        }

        for result in new_results:
            key = (result['strategy'], result['synthetic_ratio'], result['group_id'], result['trial'])
            if key not in existing_keys:
                existing_results.append(result)
                existing_keys.add(key)

        # 保存合并后的结果
        checkpoint_data = {
            'experiment_name': self.config.name,
            'summary': {
                'total_experiments': len(existing_results),
                'successful_experiments': len(existing_results),
                'failed_experiments': 0,
                'results_file': results_file
            },
            'results': existing_results,
            'config': {
                'classifiers': list(self.classifiers.keys()),
                'evaluation_metrics': self.config.evaluation.get('metrics', [])
            }
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)


# 便捷函数
def run_classification_experiments(
    config: ExperimentConfig,
    datasets_dir: str,
    output_dir: str,
    strategy: str = "within_group",
    synthetic_ratios: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    便捷函数：运行分类实验

    Args:
        config: 实验配置
        datasets_dir: 数据集目录
        output_dir: 输出目录
        strategy: 混合策略
        synthetic_ratios: synthetic比例列表

    Returns:
        Dict[str, Any]: 实验结果
    """
    if synthetic_ratios is None:
        synthetic_ratios = config.synthetic_ratios

    # 创建实验配置列表
    experiment_configs = []

    for ratio in synthetic_ratios:
        for group_id in range(config.n_groups):
            for trial in range(config.trials_per_config):
                experiment_configs.append({
                    'strategy': strategy,
                    'synthetic_ratio': ratio,
                    'group_id': group_id,
                    'trial': trial
                })

    # 运行实验
    experiment = ClassificationExperiment(config)
    return experiment.run_batch_experiments(experiment_configs, datasets_dir, output_dir)