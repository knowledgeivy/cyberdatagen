# src/cyberdata/scripts/batch3_distance_analysis.py

import pandas as pd
import numpy as np
import json
import time
import concurrent.futures
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
import pickle
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Determine PROJECT_ROOT - directly point to cyberdata root
current_file = Path(__file__).resolve()
PROJECT_ROOT = Path(r"C:\Users\Tianyu Wang\Documents\Notebooks\cyberdata")

# Verify the path exists
if not (PROJECT_ROOT / 'data').exists():
    # Try alternative locations
    potential_roots = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        current_file.parents[3]  # Original calculation
    ]
    for root in potential_roots:
        if (root / 'data').exists() and (root / 'data' / 'batch1').exists():
            PROJECT_ROOT = root
            break

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# Load environment
load_dotenv()
client = OpenAI()

# Paths
BATCH1_DIR = PROJECT_ROOT / 'data/batch1'
BATCH2_DIR = PROJECT_ROOT / 'data/batch2'
BATCH3_DIR = PROJECT_ROOT / 'data/batch3'
BATCH3_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(BATCH3_DIR / 'distance_experiments').mkdir(parents=True, exist_ok=True)
(BATCH3_DIR / 'synthetic').mkdir(parents=True, exist_ok=True)
(BATCH3_DIR / 'analysis').mkdir(parents=True, exist_ok=True)
(BATCH3_DIR / 'ml_results').mkdir(parents=True, exist_ok=True)

# Configuration
MODEL_NAME = "gpt-4o-mini"
MAX_WORKERS = 8
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Distance ranges for critical point analysis
DISTANCE_RANGES = [
    (0.30, 0.40),  # Very close - expected high quality
    (0.40, 0.50),  # Close - expected good quality
    (0.50, 0.60),  # Medium - expected moderate quality
    (0.60, 0.70),  # Far - expected quality decline
    (0.70, 0.80),  # Very far - expected poor quality
    (0.80, 0.90),  # Extremely far - expected very poor quality
    (0.90, 1.00),  # Ultra far - expected extremely poor quality
    (1.00, 1.20),  # Boundary - expected failure
]

# LLM Prompts (reuse from batch1/2)
PROMPTS = {
    'rewrite': {
        'system': "You are an expert in email phishing simulation. Rewrite phishing emails to maintain the malicious intent while changing language, structure, and specific details.",
        'user': "Rewrite this phishing email. Keep the malicious intent and core attack vector, but change the wording, structure, and specific details. Return only the rewritten email in the same format.\\n\\nOriginal email:\\nSubject: {subject}\\nBody: {body}\\n\\nRewritten email:"
    },
    'rewrite_strong': {
        'system': "You are an expert in advanced phishing email generation. Create sophisticated variants that are harder to detect while maintaining the original attack intent.",
        'user': "Create a sophisticated variant of this phishing email. Make it more convincing and harder to detect while keeping the same attack goal. Use professional language and realistic scenarios.\\n\\nOriginal email:\\nSubject: {subject}\\nBody: {body}\\n\\nSophisticated variant:"
    }
}

class Batch3DistanceAnalyzer:
    """主要的阶段3距离分析类"""
    
    def __init__(self):
        self.real_data = None
        self.existing_synthetic = None
        self.existing_embeddings = None
        self.test_set = None
        self.embedding_model = None
        self.results = []
        
    def load_data(self):
        """加载所有必要的数据"""
        print("Loading data for Batch3 analysis...")
        
        # Load real data
        self.real_data = pd.read_csv(BATCH1_DIR / 'base_samples/malicious_5k.csv')
        print(f"Loaded {len(self.real_data)} real samples")
        
        # Load existing synthetic data (batch1 + batch2)
        synthetic_files = [
            BATCH1_DIR / 'synthetic/malicious_rewrite_1k.csv',
            BATCH1_DIR / 'synthetic/malicious_rewrite_strong_1k.csv',
            BATCH1_DIR / 'synthetic/malicious_rewrite_weak_1k.csv',
            BATCH2_DIR / 'synthetic/malicious_rewrite_enhanced.csv',
            BATCH2_DIR / 'synthetic/malicious_rewrite_strong_enhanced.csv',
            BATCH2_DIR / 'synthetic/malicious_rewrite_weak_enhanced.csv',
        ]
        
        synthetic_dfs = []
        for file_path in synthetic_files:
            if file_path.exists():
                df = pd.read_csv(file_path)
                synthetic_dfs.append(df)
        
        self.existing_synthetic = pd.concat(synthetic_dfs, ignore_index=True) if synthetic_dfs else pd.DataFrame()
        print(f"Loaded {len(self.existing_synthetic)} existing synthetic samples")
        
        # Load test set
        test_file = BATCH1_DIR / 'ml_results/fixed_test_set.csv'
        if test_file.exists():
            self.test_set = pd.read_csv(test_file)
            print(f"Loaded {len(self.test_set)} test samples")
        
        # Load existing embeddings
        embeddings_file = BATCH1_DIR / 'embeddings/batch1_embeddings.pkl'
        if embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                embedding_data = pickle.load(f)
                self.existing_embeddings = embedding_data['embeddings']
                print(f"Loaded existing embeddings: {embedding_data['data_shape']}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
    def calculate_distances_to_existing(self, real_samples):
        """计算真实样本到现有合成数据的距离"""
        if self.existing_synthetic.empty or self.existing_embeddings is None:
            return np.array([])
        
        # Create text for real samples
        real_texts = (real_samples['subject'].fillna('') + ' ' + real_samples['body'].fillna('')).tolist()
        
        # Generate embeddings for real samples
        real_embeddings = self.embedding_model.encode(real_texts)
        
        # Get synthetic embeddings (assuming they're in the existing embeddings)
        # For simplicity, we'll recalculate synthetic embeddings
        synthetic_texts = (self.existing_synthetic['subject'].fillna('') + ' ' + 
                          self.existing_synthetic['body'].fillna('')).tolist()
        synthetic_embeddings = self.embedding_model.encode(synthetic_texts)
        
        # Calculate minimum distances from each real sample to synthetic data
        distances_matrix = cosine_distances(real_embeddings, synthetic_embeddings)
        min_distances = np.min(distances_matrix, axis=1)
        
        return min_distances
    
    def select_seeds_by_distance_range(self, distance_range, n_seeds=50):
        """为指定距离区间选择种子样本"""
        min_dist, max_dist = distance_range
        print(f"Selecting seeds for distance range [{min_dist:.2f}, {max_dist:.2f})")
        
        # Calculate distances
        distances = self.calculate_distances_to_existing(self.real_data)
        
        if len(distances) == 0:
            print("No distance data available")
            return []
        
        # Find samples in the target distance range
        mask = (distances >= min_dist) & (distances < max_dist)
        candidates_idx = np.where(mask)[0]
        
        if len(candidates_idx) == 0:
            print(f"No samples found in distance range [{min_dist:.2f}, {max_dist:.2f})")
            return []
        
        # Create candidate list with distances
        candidates = []
        for idx in candidates_idx:
            candidates.append({
                'index': idx,
                'sample': self.real_data.iloc[idx],
                'distance': distances[idx]
            })
        
        # Sort by distance for uniform sampling
        candidates.sort(key=lambda x: x['distance'])
        
        # Select up to n_seeds samples uniformly
        n_candidates = len(candidates)
        if n_candidates <= n_seeds:
            selected = candidates
        else:
            # Uniform sampling across the distance range
            step = n_candidates / n_seeds
            selected = [candidates[int(i * step)] for i in range(n_seeds)]
        
        print(f"Selected {len(selected)} seeds from {n_candidates} candidates")
        return selected
    
    def call_llm(self, system_prompt, user_prompt, max_retries=3):
        """调用LLM生成合成数据"""
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)
    
    def parse_email_response(self, response_text, original_subject, original_body):
        """解析LLM响应提取主题和正文"""
        try:
            lines = response_text.strip().split('\\n')
            subject = ""
            body = ""
            
            for i, line in enumerate(lines):
                if line.lower().startswith('subject:'):
                    subject = line[8:].strip()
                    body_lines = []
                    for j in range(i + 1, len(lines)):
                        if lines[j].lower().startswith('body:'):
                            body_lines = lines[j + 1:]
                            break
                        elif not lines[j].strip():
                            continue
                        else:
                            body_lines = lines[i + 1:]
                            break
                    body = '\\n'.join(body_lines).strip()
                    break
            
            # Alternative parsing if needed
            if not subject and 'subject:' in response_text.lower():
                parts = response_text.split('Subject:', 1)
                if len(parts) > 1:
                    remaining = parts[1]
                    if 'Body:' in remaining:
                        subject_part, body_part = remaining.split('Body:', 1)
                        subject = subject_part.strip()
                        body = body_part.strip()
                    else:
                        lines = remaining.strip().split('\\n')
                        subject = lines[0].strip() if lines else ""
                        body = '\\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
            
            # Fallback to original if parsing failed
            if not subject:
                subject = original_subject
            if not body:
                body = original_body
                
            return subject, body
        except Exception:
            return original_subject, original_body
    
    def generate_synthetic_from_seeds(self, seeds, distance_range, variants=['rewrite', 'rewrite_strong']):
        """从种子样本生成合成数据"""
        min_dist, max_dist = distance_range
        range_name = f"{min_dist:.2f}_{max_dist:.2f}".replace('.', 'p')
        
        print(f"Generating synthetic data for distance range [{min_dist:.2f}, {max_dist:.2f})")
        
        all_synthetic = []
        
        for variant in variants:
            print(f"  Generating {variant} variants...")
            
            variant_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []
                
                for i, seed_info in enumerate(seeds):
                    seed_sample = seed_info['sample']
                    future = executor.submit(self.generate_single_sample, seed_sample, variant, i, range_name)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    variant_results.append(result)
            
            # Convert to DataFrame and save
            variant_df = pd.DataFrame(variant_results)
            variant_file = BATCH3_DIR / 'synthetic' / f'malicious_{variant}_range_{range_name}.csv'
            variant_df.to_csv(variant_file, index=False)
            
            all_synthetic.extend(variant_results)
            
            success_count = variant_df['generation_success'].sum()
            print(f"    Generated {len(variant_df)} samples ({success_count} successful)")
        
        return pd.DataFrame(all_synthetic)
    
    def generate_single_sample(self, seed_sample, variant_type, sample_idx, range_name):
        """生成单个合成样本"""
        try:
            original_subject = str(seed_sample['subject'])
            original_body = str(seed_sample['body'])
            original_id = seed_sample['data_id']
            
            prompt_config = PROMPTS[variant_type]
            system_prompt = prompt_config['system']
            user_prompt = prompt_config['user'].format(
                subject=original_subject,
                body=original_body
            )
            
            response = self.call_llm(system_prompt, user_prompt)
            subject, body = self.parse_email_response(response, original_subject, original_body)
            
            # Generate synthetic ID for batch3
            synthetic_id = f"CEAS08_SYN3_{variant_type.upper()[:2]}_{range_name}_{sample_idx:03d}"
            
            return {
                'data_id': synthetic_id,
                'subject': subject,
                'body': body,
                'label': 1,
                'original_id': original_id,
                'variant_type': variant_type,
                'distance_range': range_name,
                'generation_success': True
            }
        except Exception as e:
            return {
                'data_id': f"FAILED_B3_{variant_type}_{range_name}_{sample_idx}",
                'subject': seed_sample['subject'],
                'body': seed_sample['body'],
                'label': 1,
                'original_id': seed_sample['data_id'],
                'variant_type': variant_type,
                'distance_range': range_name,
                'generation_success': False,
                'error': str(e)
            }
    
    def evaluate_classification_performance(self, training_data, test_data):
        """评估分类性能"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Prepare training data
        train_texts = (training_data['subject'].fillna('') + ' ' + training_data['body'].fillna('')).values
        train_labels = training_data['label'].values
        
        # Prepare test data  
        test_texts = (test_data['subject'].fillna('') + ' ' + test_data['body'].fillna('')).values
        test_labels = test_data['label'].values
        
        # Check for single-class issues
        unique_train_labels = np.unique(train_labels)
        unique_test_labels = np.unique(test_labels)
        
        print(f"Training classes: {unique_train_labels}, Test classes: {unique_test_labels}")
        
        if len(unique_train_labels) < 2:
            print(f"Warning: Training data has only {len(unique_train_labels)} class(es)")
            # Return default scores for single-class case
            return {
                'RandomForest': {
                    'accuracy': 0.5,
                    'f1_macro': 0.0,
                    'f1_weighted': 0.0,
                    'auc_roc': 0.5
                },
                'SVM': {
                    'accuracy': 0.5,
                    'f1_macro': 0.0,
                    'f1_weighted': 0.0,
                    'auc_roc': 0.5
                }
            }
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)
        
        # Train and evaluate models
        results = {}
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, train_labels)
        rf_pred = rf.predict(X_test)
        
        # Handle single-class prediction probability
        rf_proba_full = rf.predict_proba(X_test)
        if rf_proba_full.shape[1] > 1:
            rf_proba = rf_proba_full[:, 1]
        else:
            # Single class case - use the only available probability
            rf_proba = rf_proba_full[:, 0]
        
        results['RandomForest'] = {
            'accuracy': accuracy_score(test_labels, rf_pred),
            'f1_macro': f1_score(test_labels, rf_pred, average='macro'),
            'f1_weighted': f1_score(test_labels, rf_pred, average='weighted'),
            'auc_roc': roc_auc_score(test_labels, rf_proba) if len(np.unique(test_labels)) > 1 else 0.0
        }
        
        # SVM
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train, train_labels)
        svm_pred = svm.predict(X_test)
        
        # Handle single-class prediction probability
        svm_proba_full = svm.predict_proba(X_test)
        if svm_proba_full.shape[1] > 1:
            svm_proba = svm_proba_full[:, 1]
        else:
            # Single class case - use the only available probability
            svm_proba = svm_proba_full[:, 0]
        
        results['SVM'] = {
            'accuracy': accuracy_score(test_labels, svm_pred),
            'f1_macro': f1_score(test_labels, svm_pred, average='macro'),
            'f1_weighted': f1_score(test_labels, svm_pred, average='weighted'),
            'auc_roc': roc_auc_score(test_labels, svm_proba) if len(np.unique(test_labels)) > 1 else 0.0
        }
        
        return results
    
    def run_single_distance_experiment(self, distance_range):
        """运行单个距离区间的完整实验"""
        min_dist, max_dist = distance_range
        print(f"\\n=== Running experiment for distance range [{min_dist:.2f}, {max_dist:.2f}) ===")
        
        # Step 1: Select seeds
        seeds = self.select_seeds_by_distance_range(distance_range, n_seeds=50)
        
        if len(seeds) == 0:
            print("No seeds available for this distance range")
            return None
        
        # Step 2: Generate synthetic data
        synthetic_data = self.generate_synthetic_from_seeds(seeds, distance_range)
        
        if len(synthetic_data) == 0:
            print("No synthetic data generated")
            return None
        
        # Step 3: Combine with existing data for training
        # Include both synthetic data (malicious) and benign data for balanced training
        training_data_malicious = pd.concat([self.existing_synthetic, synthetic_data], ignore_index=True)
        
        # Load benign data for balanced training
        benign_file = BATCH1_DIR / 'base_samples/benign_5k.csv'
        if benign_file.exists():
            benign_data = pd.read_csv(benign_file)
            # Sample same amount of benign data as malicious for balance
            n_malicious = len(training_data_malicious)
            if len(benign_data) >= n_malicious:
                sampled_benign = benign_data.sample(n=n_malicious, random_state=42)
            else:
                sampled_benign = benign_data
            
            training_data = pd.concat([training_data_malicious, sampled_benign], ignore_index=True)
            print(f"Training data: {len(training_data_malicious)} malicious + {len(sampled_benign)} benign = {len(training_data)} total")
        else:
            training_data = training_data_malicious
            print(f"Warning: No benign data found, using only malicious data ({len(training_data)} samples)")
        
        # Step 4: Evaluate performance
        if self.test_set is not None:
            performance = self.evaluate_classification_performance(training_data, self.test_set)
        else:
            performance = {}
        
        # Step 5: Calculate quality metrics
        quality_metrics = {
            'n_seeds': len(seeds),
            'n_generated': len(synthetic_data),
            'generation_success_rate': synthetic_data['generation_success'].mean() if len(synthetic_data) > 0 else 0.0,
            'avg_seed_distance': np.mean([s['distance'] for s in seeds]),
            'distance_std': np.std([s['distance'] for s in seeds])
        }
        
        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        result = {
            'distance_range': distance_range,
            'range_name': f"{min_dist:.2f}_{max_dist:.2f}".replace('.', 'p'),
            'performance': convert_numpy_types(performance),
            'quality_metrics': convert_numpy_types(quality_metrics),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save individual experiment result
        result_file = BATCH3_DIR / 'distance_experiments' / f'experiment_{result["range_name"]}.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def detect_critical_points(self, results, threshold=0.05):
        """检测性能急剧下降的临界点"""
        if len(results) < 2:
            return []
        
        critical_points = []
        
        # Sort results by distance
        sorted_results = sorted(results, key=lambda x: x['distance_range'][0])
        
        for i in range(1, len(sorted_results)):
            prev_result = sorted_results[i-1]
            curr_result = sorted_results[i]
            
            # Check both RandomForest and SVM
            for model in ['RandomForest', 'SVM']:
                if model in prev_result['performance'] and model in curr_result['performance']:
                    prev_acc = prev_result['performance'][model]['accuracy']
                    curr_acc = curr_result['performance'][model]['accuracy']
                    
                    performance_drop = prev_acc - curr_acc
                    
                    if performance_drop > threshold:
                        critical_points.append({
                            'distance_range': curr_result['distance_range'],
                            'model': model,
                            'performance_drop': performance_drop,
                            'previous_accuracy': prev_acc,
                            'current_accuracy': curr_acc,
                            'critical_score': performance_drop / threshold  # Severity score
                        })
        
        return critical_points
    
    def run_full_analysis(self):
        """运行完整的阶段3分析"""
        print("=== BATCH 3: DISTANCE CRITICAL POINT ANALYSIS ===")
        
        # Load all data
        self.load_data()
        
        if self.real_data is None or self.existing_synthetic.empty:
            print("Error: Required data not available")
            return None
        
        # Run experiments for all distance ranges
        results = []
        for distance_range in DISTANCE_RANGES:
            result = self.run_single_distance_experiment(distance_range)
            if result is not None:
                results.append(result)
                self.results.append(result)
        
        # Detect critical points
        critical_points = self.detect_critical_points(results)
        
        # Save overall results
        overall_results = {
            'experiment': 'batch3_distance_analysis',
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if r is not None]),
            'distance_ranges_tested': DISTANCE_RANGES,
            'critical_points': critical_points,
            'individual_results': results
        }
        
        with open(BATCH3_DIR / 'batch3_analysis_results.json', 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        print(f"\\nAnalysis complete:")
        print(f"- Tested {len(results)} distance ranges")
        print(f"- Found {len(critical_points)} critical points")
        print(f"- Results saved to: {BATCH3_DIR}")
        
        return overall_results

def main():
    """主函数"""
    analyzer = Batch3DistanceAnalyzer()
    results = analyzer.run_full_analysis()
    return results

if __name__ == "__main__":
    main()