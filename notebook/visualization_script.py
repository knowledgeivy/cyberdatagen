import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import json
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class EmailDataset(Dataset):
    """Custom dataset for email data"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class SyntheticDataVisualizer:
    """Visualize synthetic cybersecurity data using transformer embeddings and t-SNE"""
    
    def __init__(self, model_name='distilbert-base-uncased'):
        """Initialize the visualizer with a transformer model"""
        print(f"Loading model: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_real_data(self, file_path, sample_size=None):
        """Load real-world data for comparison"""
        print(f"Loading real data from: {file_path}")
        
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Real data shape: {df.shape}")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = self._stratified_sample(df, sample_size)
        
        # Add data source column
        df['data_source'] = 'Real'
        
        return df
    
    def load_synthetic_data(self, file_path, sample_size=None):
        """Load synthetic data from CyberData JSON format"""
        print(f"Loading synthetic data from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract samples from CyberData format
        if 'samples' in data:
            samples = data['samples']
        elif 'examples' in data:
            samples = data['examples']
        else:
            raise ValueError("Could not find 'samples' or 'examples' in synthetic data file")
        
        df = pd.DataFrame(samples)
        print(f"Synthetic data shape: {df.shape}")
        print(f"Synthetic data columns: {df.columns.tolist()}")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = self._stratified_sample(df, sample_size)
        
        # Add data source column
        df['data_source'] = 'Synthetic'
        
        return df
    
    def load_multiple_synthetic_files(self, synthetic_dir, pattern="*_scale.json", sample_size_per_file=None):
        """Load multiple synthetic data files from a directory"""
        synthetic_dir = Path(synthetic_dir)
        synthetic_files = list(synthetic_dir.rglob(pattern))
        
        print(f"Found {len(synthetic_files)} synthetic files in {synthetic_dir}")
        
        all_dfs = []
        for i, file_path in enumerate(synthetic_files):
            print(f"Loading {file_path.name}...")
            try:
                df = self.load_synthetic_data(file_path, sample_size_per_file)
                df['synthetic_source'] = file_path.stem  # Add file source
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("No synthetic files could be loaded")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined synthetic data shape: {combined_df.shape}")
        
        return combined_df
    
    def _stratified_sample(self, df, sample_size):
        """Perform stratified sampling maintaining label distribution"""
        if 'label' not in df.columns:
            return df.sample(n=min(sample_size, len(df)), random_state=42)
        
        label_dist = df['label'].value_counts()
        total_samples = len(df)
        
        sampled_dfs = []
        for label_value, count in label_dist.items():
            label_ratio = count / total_samples
            target_samples = max(1, int(sample_size * label_ratio))
            target_samples = min(target_samples, count)
            
            label_df = df[df['label'] == label_value].sample(n=target_samples, random_state=42)
            sampled_dfs.append(label_df)
        
        return pd.concat(sampled_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    def prepare_text_from_dataframe(self, df, text_columns=None):
        """Prepare text data from DataFrame with flexible column handling"""
        
        if text_columns is None:
            # Auto-detect text columns based on common CyberData schema
            possible_columns = ['subject', 'body', 'text', 'content', 'message', 'description']
            text_columns = [col for col in possible_columns if col in df.columns]
        
        if not text_columns:
            # Fallback: use all string columns except metadata
            exclude_cols = ['label', 'Label', 'data_source', 'synthetic_source', 'sample_id', 
                          'generation_timestamp', '_metadata']
            text_columns = [col for col in df.columns if df[col].dtype == 'object' and col not in exclude_cols]
        
        print(f"Using text columns: {text_columns}")
        
        if not text_columns:
            raise ValueError("No suitable text columns found in the data")
        
        # Combine text columns
        texts = []
        for _, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            
            combined_text = " [SEP] ".join(text_parts) if text_parts else ""
            texts.append(combined_text[:2000])  # Limit length
        
        print(f"Prepared {len(texts)} text samples")
        print(f"Average text length: {np.mean([len(t) for t in texts]):.1f} characters")
        
        return texts
    
    def get_embeddings(self, texts, batch_size=16, max_length=512):
        """Get transformer embeddings for texts"""
        print(f"Generating embeddings for {len(texts)} texts...")
        
        dataset = EmailDataset(texts, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i % 10 == 0:
                    print(f"Processing batch {i+1}/{len(dataloader)}")
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def apply_tsne(self, embeddings, n_components=2, perplexity=30, random_state=42):
        """Apply t-SNE dimensionality reduction"""
        print(f"Applying t-SNE (perplexity={perplexity})...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(embeddings) - 1),
            random_state=random_state,
            max_iter=1000,
            verbose=1
        )
        
        embeddings_2d = tsne.fit_transform(embeddings_scaled)
        
        print(f"t-SNE completed. Output shape: {embeddings_2d.shape}")
        return embeddings_2d
    
    def visualize_synthetic_comparison(self, embeddings_2d, combined_df, save_path=None):
        """Create comprehensive visualization comparing real and synthetic data"""
        print("Creating synthetic data comparison visualizations...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Real vs Synthetic (top left)
        ax1 = plt.subplot(2, 3, 1)
        
        real_mask = combined_df['data_source'] == 'Real'
        synthetic_mask = combined_df['data_source'] == 'Synthetic'
        
        ax1.scatter(
            embeddings_2d[real_mask, 0], 
            embeddings_2d[real_mask, 1],
            c='navy', alpha=0.6, s=20, label=f'Real ({np.sum(real_mask)})',
            edgecolors='black', linewidth=0.1
        )
        
        ax1.scatter(
            embeddings_2d[synthetic_mask, 0], 
            embeddings_2d[synthetic_mask, 1],
            c='orange', alpha=0.6, s=20, label=f'Synthetic ({np.sum(synthetic_mask)})',
            edgecolors='black', linewidth=0.1
        )
        
        ax1.set_title('Real vs Synthetic Data', fontsize=12, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Labels in Real Data (top middle)
        ax2 = plt.subplot(2, 3, 2)
        
        real_data = combined_df[real_mask]
        real_embeddings = embeddings_2d[real_mask]
        
        if 'label' in real_data.columns:
            benign_mask_real = real_data['label'] == 0
            malicious_mask_real = real_data['label'] == 1
            
            ax2.scatter(
                real_embeddings[benign_mask_real, 0], 
                real_embeddings[benign_mask_real, 1],
                c='lightblue', alpha=0.6, s=20, 
                label=f'Real Benign ({np.sum(benign_mask_real)})',
                edgecolors='navy', linewidth=0.1
            )
            
            ax2.scatter(
                real_embeddings[malicious_mask_real, 0], 
                real_embeddings[malicious_mask_real, 1],
                c='red', alpha=0.7, s=20, 
                label=f'Real Malicious ({np.sum(malicious_mask_real)})',
                edgecolors='darkred', linewidth=0.1
            )
        
        ax2.set_title('Real Data Labels', fontsize=12, fontweight='bold')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Labels in Synthetic Data (top right)
        ax3 = plt.subplot(2, 3, 3)
        
        synthetic_data = combined_df[synthetic_mask]
        synthetic_embeddings = embeddings_2d[synthetic_mask]
        
        if 'label' in synthetic_data.columns:
            benign_mask_syn = synthetic_data['label'] == 0
            malicious_mask_syn = synthetic_data['label'] == 1
            
            ax3.scatter(
                synthetic_embeddings[benign_mask_syn, 0], 
                synthetic_embeddings[benign_mask_syn, 1],
                c='lightgreen', alpha=0.6, s=20, 
                label=f'Synthetic Benign ({np.sum(benign_mask_syn)})',
                edgecolors='green', linewidth=0.1
            )
            
            ax3.scatter(
                synthetic_embeddings[malicious_mask_syn, 0], 
                synthetic_embeddings[malicious_mask_syn, 1],
                c='purple', alpha=0.7, s=20, 
                label=f'Synthetic Malicious ({np.sum(malicious_mask_syn)})',
                edgecolors='indigo', linewidth=0.1
            )
        
        ax3.set_title('Synthetic Data Labels', fontsize=12, fontweight='bold')
        ax3.set_xlabel('t-SNE Dimension 1')
        ax3.set_ylabel('t-SNE Dimension 2')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined Labels (bottom left)
        ax4 = plt.subplot(2, 3, 4)
        
        if 'label' in combined_df.columns:
            # Real data points
            real_benign = real_mask & (combined_df['label'] == 0)
            real_malicious = real_mask & (combined_df['label'] == 1)
            
            # Synthetic data points
            syn_benign = synthetic_mask & (combined_df['label'] == 0)
            syn_malicious = synthetic_mask & (combined_df['label'] == 1)
            
            ax4.scatter(
                embeddings_2d[real_benign, 0], embeddings_2d[real_benign, 1],
                c='lightblue', alpha=0.6, s=20, label='Real Benign',
                edgecolors='navy', linewidth=0.1, marker='o'
            )
            ax4.scatter(
                embeddings_2d[real_malicious, 0], embeddings_2d[real_malicious, 1],
                c='red', alpha=0.7, s=20, label='Real Malicious',
                edgecolors='darkred', linewidth=0.1, marker='o'
            )
            ax4.scatter(
                embeddings_2d[syn_benign, 0], embeddings_2d[syn_benign, 1],
                c='lightgreen', alpha=0.6, s=20, label='Synthetic Benign',
                edgecolors='green', linewidth=0.1, marker='^'
            )
            ax4.scatter(
                embeddings_2d[syn_malicious, 0], embeddings_2d[syn_malicious, 1],
                c='purple', alpha=0.7, s=20, label='Synthetic Malicious',
                edgecolors='indigo', linewidth=0.1, marker='^'
            )
        
        ax4.set_title('Combined: Real vs Synthetic Labels', fontsize=12, fontweight='bold')
        ax4.set_xlabel('t-SNE Dimension 1')
        ax4.set_ylabel('t-SNE Dimension 2')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Synthetic Sources (bottom middle)
        ax5 = plt.subplot(2, 3, 5)
        
        if 'synthetic_source' in combined_df.columns:
            synthetic_data_only = combined_df[synthetic_mask]
            synthetic_embeddings_only = embeddings_2d[synthetic_mask]
            
            unique_sources = synthetic_data_only['synthetic_source'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))
            
            for i, source in enumerate(unique_sources):
                source_mask = synthetic_data_only['synthetic_source'] == source
                ax5.scatter(
                    synthetic_embeddings_only[source_mask, 0],
                    synthetic_embeddings_only[source_mask, 1],
                    c=[colors[i]], alpha=0.6, s=20,
                    label=f'{source} ({np.sum(source_mask)})',
                    edgecolors='black', linewidth=0.1
                )
        
        ax5.set_title('Synthetic Data Sources', fontsize=12, fontweight='bold')
        ax5.set_xlabel('t-SNE Dimension 1')
        ax5.set_ylabel('t-SNE Dimension 2')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Quality Distribution (bottom right)
        ax6 = plt.subplot(2, 3, 6)
        
        # Create quality heatmap or density plot
        from scipy.stats import gaussian_kde
        
        try:
            if len(embeddings_2d) > 100:
                # Create density plot
                xy = embeddings_2d.T
                density = gaussian_kde(xy)(xy)
                
                scatter = ax6.scatter(
                    embeddings_2d[:, 0], embeddings_2d[:, 1],
                    c=density, s=20, alpha=0.6, cmap='viridis'
                )
                plt.colorbar(scatter, ax=ax6, label='Density')
        except:
            # Fallback to simple scatter
            ax6.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       c='gray', alpha=0.5, s=20)
        
        ax6.set_title('Data Density Distribution', fontsize=12, fontweight='bold')
        ax6.set_xlabel('t-SNE Dimension 1')
        ax6.set_ylabel('t-SNE Dimension 2')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def print_data_statistics(self, combined_df):
        """Print comprehensive statistics about the data"""
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        
        print(f"Total samples: {len(combined_df)}")
        
        # Data source distribution
        source_dist = combined_df['data_source'].value_counts()
        print(f"\nData Source Distribution:")
        for source, count in source_dist.items():
            print(f"  {source}: {count} ({count/len(combined_df)*100:.1f}%)")
        
        # Label distribution by source
        if 'label' in combined_df.columns:
            print(f"\nLabel Distribution by Source:")
            for source in combined_df['data_source'].unique():
                source_data = combined_df[combined_df['data_source'] == source]
                label_dist = source_data['label'].value_counts()
                print(f"  {source}:")
                for label, count in label_dist.items():
                    label_name = "Malicious" if label == 1 else "Benign"
                    print(f"    {label_name}: {count} ({count/len(source_data)*100:.1f}%)")
        
        # Synthetic source distribution
        if 'synthetic_source' in combined_df.columns:
            syn_data = combined_df[combined_df['data_source'] == 'Synthetic']
            if len(syn_data) > 0:
                syn_source_dist = syn_data['synthetic_source'].value_counts()
                print(f"\nSynthetic Source Distribution:")
                for source, count in syn_source_dist.items():
                    print(f"  {source}: {count} ({count/len(syn_data)*100:.1f}%)")

def main():
    """Main function to run synthetic data visualization"""
    
    # Configuration - Update these paths for your setup
    REAL_DATA_FILE = "../raw/five_email_phishing_train.csv.gz"
    SYNTHETIC_DATA_DIR = "../data/scaled-validated"  # Directory with synthetic JSON files
    # OR use a specific file:
    # SYNTHETIC_DATA_FILE = "../data/scaled-validated/Social_Engineering/phishing_scale.json"
    
    MODEL_NAME = "distilbert-base-uncased"
    REAL_SAMPLE_SIZE = 1000  # Limit real data for faster processing
    SYNTHETIC_SAMPLE_SIZE = 1000  # Limit synthetic data
    SAVE_PATH = "synthetic_data_comparison.png"
    
    print("="*60)
    print("Synthetic Cybersecurity Data Visualization")
    print("="*60)
    
    # Initialize visualizer
    visualizer = SyntheticDataVisualizer(model_name=MODEL_NAME)
    
    # Load real data
    real_df = visualizer.load_real_data(REAL_DATA_FILE, sample_size=REAL_SAMPLE_SIZE)
    
    # Load synthetic data
    # Option 1: Load from directory (multiple files)
    try:
        synthetic_df = visualizer.load_multiple_synthetic_files(
            SYNTHETIC_DATA_DIR, 
            pattern="*_scale.json",
            sample_size_per_file=SYNTHETIC_SAMPLE_SIZE//2  # Adjust based on number of files
        )
    except:
        print("Could not load from directory, trying single file...")
        # Option 2: Load single synthetic file
        synthetic_file = Path(SYNTHETIC_DATA_DIR) / "Social_Engineering" / "phishing_scale.json"
        synthetic_df = visualizer.load_synthetic_data(synthetic_file, sample_size=SYNTHETIC_SAMPLE_SIZE)
    
    # Combine datasets
    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    
    # Print statistics
    visualizer.print_data_statistics(combined_df)
    
    # Prepare text data
    texts = visualizer.prepare_text_from_dataframe(combined_df)
    
    # Get embeddings
    embeddings = visualizer.get_embeddings(texts, batch_size=8)
    
    # Apply t-SNE
    embeddings_2d = visualizer.apply_tsne(embeddings, perplexity=30)
    
    # Create visualizations
    visualizer.visualize_synthetic_comparison(
        embeddings_2d, 
        combined_df, 
        save_path=SAVE_PATH
    )
    
    print(f"\nVisualization completed!")
    print(f"Comparison plot saved as: {SAVE_PATH}")

if __name__ == "__main__":
    main()