import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')  # More compatible than seaborn-v0_8
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

class PhishingDataVisualizer:
    """Visualize phishing email data using PhishBERT embeddings and t-SNE"""
    
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        """
        Initialize the visualizer
        
        Args:
            model_name: HuggingFace model name. You can use:
                - 'microsoft/DialoGPT-medium' (good for general text)
                - 'distilbert-base-uncased' (lighter, faster)
                - 'bert-base-uncased' (classic BERT)
                - Or any other transformer model
        """
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
    
    def load_data(self, file_path, sample_size=None):
        """
        Load email phishing data
        
        Args:
            file_path: Path to the CSV file
            sample_size: Number of samples to use (None for all)
        """
        print(f"Loading data from: {file_path}")
        
        # Load data (handle both .gz and regular files)
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check label distribution
        label_dist = df['label'].value_counts()
        print(f"\nLabel distribution:")
        print(f"  Benign (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Malicious (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # Sample data if requested
        if sample_size and len(df) > sample_size:
            # Stratified sampling to maintain label distribution
            malicious_ratio = label_dist.get(1, 0) / len(df)
            malicious_samples = int(sample_size * malicious_ratio)
            benign_samples = sample_size - malicious_samples
            
            malicious_df = df[df['label'] == 1].sample(n=min(malicious_samples, len(df[df['label'] == 1])), random_state=42)
            benign_df = df[df['label'] == 0].sample(n=min(benign_samples, len(df[df['label'] == 0])), random_state=42)
            
            df = pd.concat([malicious_df, benign_df]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Sampled to {len(df)} examples")
        
        return df
    
    def prepare_text(self, df, text_column=None):
        """
        Prepare text data for embedding
        
        Args:
            df: DataFrame with email data
            text_column: Column to use for text (None to combine subject and body)
        """
        if text_column:
            # Use specific column
            texts = df[text_column].fillna('').astype(str)
        else:
            # Combine subject and body
            subjects = df['subject'].fillna('').astype(str)
            bodies = df['body'].fillna('').astype(str)
            # Combine with separator
            texts = subjects + " [SEP] " + bodies
        
        # Clean and prepare texts
        texts = texts.apply(lambda x: x.strip()[:2000])  # Limit length
        
        print(f"Prepared {len(texts)} text samples")
        print(f"Average text length: {texts.str.len().mean():.1f} characters")
        
        return texts.tolist()
    
    def get_embeddings(self, texts, batch_size=16, max_length=512):
        """
        Get PhishBERT embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
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
                
                # Get model outputs
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Use [CLS] token embedding (first token) or mean pooling
                # Option 1: [CLS] token
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Option 2: Mean pooling (often better)
                # masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                # cls_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                embeddings.append(cls_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def apply_tsne(self, embeddings, n_components=2, perplexity=30, random_state=42):
        """
        Apply t-SNE dimensionality reduction
        
        Args:
            embeddings: High-dimensional embeddings
            n_components: Number of dimensions for t-SNE (2 or 3)
            perplexity: t-SNE perplexity parameter
            random_state: Random seed
        """
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
    
    def visualize_results(self, embeddings_2d, labels, sources=None, save_path=None):
        """
        Create visualization plots
        
        Args:
            embeddings_2d: 2D t-SNE embeddings
            labels: Binary labels (0=benign, 1=malicious)
            sources: Source information for coloring
            save_path: Path to save the plot
        """
        print("Creating visualizations...")
        
        # Create figure with subplots
        if sources is not None:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            axes = [ax]
        
        # Plot 1: Color by malicious/benign labels
        ax1 = axes[0]
        
        # Create scatter plot
        benign_mask = labels == 0
        malicious_mask = labels == 1
        
        scatter1 = ax1.scatter(
            embeddings_2d[benign_mask, 0], 
            embeddings_2d[benign_mask, 1],
            c='lightblue', 
            alpha=0.6, 
            s=20, 
            label=f'Benign ({np.sum(benign_mask)})',
            edgecolors='navy',
            linewidth=0.1
        )
        
        scatter2 = ax1.scatter(
            embeddings_2d[malicious_mask, 0], 
            embeddings_2d[malicious_mask, 1],
            c='red', 
            alpha=0.7, 
            s=20, 
            label=f'Malicious ({np.sum(malicious_mask)})',
            edgecolors='darkred',
            linewidth=0.1
        )
        
        ax1.set_title('Email Classification: t-SNE Visualization\n(PhishBERT Embeddings)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Color by source (if available)
        if sources is not None and len(axes) > 1:
            ax2 = axes[1]
            
            unique_sources = np.unique(sources)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))
            
            for i, source in enumerate(unique_sources):
                source_mask = sources == source
                ax2.scatter(
                    embeddings_2d[source_mask, 0],
                    embeddings_2d[source_mask, 1],
                    c=[colors[i]],
                    alpha=0.6,
                    s=20,
                    label=f'{source} ({np.sum(source_mask)})',
                    edgecolors='black',
                    linewidth=0.1
                )
            
            ax2.set_title('Email Sources: t-SNE Visualization\n(PhishBERT Embeddings)', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self, embeddings_2d, labels, texts, sources=None):
        """
        Create an interactive plot using plotly (if available)
        """
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Prepare data
            df_plot = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'label': ['Malicious' if l == 1 else 'Benign' for l in labels],
                'text_preview': [str(t)[:100] + '...' if len(str(t)) > 100 else str(t) for t in texts]
            })
            
            if sources is not None:
                df_plot['source'] = sources
            
            # Create interactive scatter plot
            fig = px.scatter(
                df_plot, 
                x='x', 
                y='y', 
                color='label',
                color_discrete_map={'Benign': 'lightblue', 'Malicious': 'red'},
                hover_data=['text_preview'],
                title='Interactive t-SNE Visualization of Email Data (PhishBERT Embeddings)',
                labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'}
            )
            
            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(width=800, height=600)
            
            return fig
            
        except ImportError:
            print("Plotly not available. Install with: pip install plotly")
            return None

def main():
    """Main function to run the visualization"""
    
    # Configuration
    TRAIN_FILE = "../raw/five_email_phishing_train.csv.gz"
    MODEL_NAME = "distilbert-base-uncased"  # Fast and effective
    SAMPLE_SIZE = 2000  # Adjust based on your computational resources
    SAVE_PATH = "phishing_tsne_visualization.png"
    
    # Alternative models you can try:
    # MODEL_NAME = "bert-base-uncased"  # Classic BERT
    # MODEL_NAME = "microsoft/DialoGPT-medium"  # Good for conversational text
    # MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Optimized for sentence embeddings
    
    print("="*60)
    print("Email Phishing Data Visualization with PhishBERT and t-SNE")
    print("="*60)
    
    # Initialize visualizer
    visualizer = PhishingDataVisualizer(model_name=MODEL_NAME)
    
    # Load data
    df = visualizer.load_data(TRAIN_FILE, sample_size=SAMPLE_SIZE)
    
    # Prepare text (combine subject and body)
    texts = visualizer.prepare_text(df)
    
    # Get embeddings
    embeddings = visualizer.get_embeddings(texts, batch_size=8)  # Reduce batch_size if you get memory errors
    
    # Apply t-SNE
    embeddings_2d = visualizer.apply_tsne(embeddings, perplexity=30)
    
    # Create visualizations
    labels = df['label'].values
    sources = df['source'].values if 'source' in df.columns else None
    
    visualizer.visualize_results(
        embeddings_2d, 
        labels, 
        sources=sources,
        save_path=SAVE_PATH
    )
    
    # Create interactive plot if possible
    interactive_fig = visualizer.create_interactive_plot(
        embeddings_2d, 
        labels, 
        texts, 
        sources=sources
    )
    
    if interactive_fig:
        interactive_fig.show()
    
    print("\nVisualization completed!")
    print(f"Static plot saved as: {SAVE_PATH}")

if __name__ == "__main__":
    main()