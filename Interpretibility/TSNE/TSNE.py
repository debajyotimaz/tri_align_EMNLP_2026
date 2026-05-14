"""
Visualize sentence representations from mBERT and Hing-mBERT-Mixed
to compare how EN, HI, and CM sentences cluster in embedding space.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - colorblind friendly
COLORS = {
    'EN': '#2E86AB',   # Blue
    'HI': '#A23B72',   # Magenta/Purple
    'CM': '#F18F01',   # Orange
}

MARKERS = {
    'EN': 'o',
    'HI': 's',
    'CM': '^',
}

# Model name mapping for display
NAME_MAP = {
    'mbert': 'mBERT',
    # 'hing_mbert': 'Hing-mBERT',
    # 'hing_mbert_mixed': 'Hing-mBERT-Mixed',
    'xlm_roberta_base': 'XLM-R Base',
    # 'hing_roberta': 'Hing-RoBERTa',
    # 'hing_roberta_mixed': 'Hing-RoBERTa-Mixed',
    'mbert_trilingual_aligned': 'mBERT-Tri-aligned',
    'mbert_ablation': 'mBERT\n–w/o Alignment',
    'xlmr_trilingual_aligned': 'XLM-R-Tri-aligned',
    'xlmr_ablation': 'XLM-R\n–w/o Alignment',
}

# Base directory for representations
BASE_REP_DIR = "/rep-path/rep_cleaned"

# Define model groups
BERT_MODELS = [
    'mbert',
    # 'mbert_trilingual_aligned',
    # 'hindi_bert_v2',
    # 'hing_mbert',
    # 'hing_mbert_mixed',
    # 'rembert'
    'mbert_trilingual_aligned',
    'mbert_ablation'
]

ROBERTA_MODELS = [
    # 'hindi_roberta',
    'xlm_roberta_base',
    'xlmr_trilingual_aligned',
    'xlmr_ablation'
    # 'hing_roberta',
    # 'hing_roberta_mixed',
    # 'xlm_roberta_base',
    # 'xlm_roberta_large'
]

# Global sample indices for consistency
SAMPLE_INDICES = None


def get_sample_indices(total_size, n_samples, random_state=42):
    """Get or create consistent sample indices."""
    global SAMPLE_INDICES
    if SAMPLE_INDICES is None or len(SAMPLE_INDICES) != n_samples:
        np.random.seed(random_state)
        SAMPLE_INDICES = np.random.choice(total_size, n_samples, replace=False)
    return SAMPLE_INDICES


def load_representations(rep_dir, layer_idx, n_samples=None, random_state=42):
    """Load pre-computed representations for a specific layer."""
    embeddings = {}
    
    # Map language keys to folder names
    lang_map = {
        'EN': 'en',
        'HI': 'hi', 
        'CM': 'cm'
    }
    
    for lang_key, folder_name in lang_map.items():
        file_path = os.path.join(rep_dir, folder_name, f"layer_{layer_idx:02d}.npy")
        data = np.load(file_path)
        
        # Sample if requested
        if n_samples is not None and n_samples < len(data):
            sample_indices = get_sample_indices(len(data), n_samples, random_state)
            data = data[sample_indices]
        
        embeddings[lang_key] = data
        print(f"  Loaded {lang_key}: {embeddings[lang_key].shape}")
    
    return embeddings


def reduce_dimensions(embeddings_dict, method='tsne', perplexity=30, random_state=42):
    """
    Reduce dimensionality of embeddings using t-SNE or PCA.
    """
    # Combine all embeddings
    all_embeddings = np.vstack([embeddings_dict[lang] for lang in ['EN', 'HI', 'CM']])
    
    print(f"  Total samples: {all_embeddings.shape[0]}, Dim: {all_embeddings.shape[1]}")
    
    if method == 'tsne':
        # First reduce with PCA to 50 dims, then t-SNE (faster & more stable)
        print("  PCA preprocessing...")
        pca = PCA(n_components=50, random_state=random_state)
        all_embeddings = pca.fit_transform(all_embeddings)
        
        print("  Running t-SNE...")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                       n_iter=1000, init='random', learning_rate='auto', method='barnes_hut')
    else:
        reducer = PCA(n_components=2, random_state=random_state)
    
    reduced = reducer.fit_transform(all_embeddings)
    
    # Clean up
    del all_embeddings
    gc.collect()
    
    # Split back
    result = {}
    idx = 0
    for lang in ['EN', 'HI', 'CM']:
        n = len(embeddings_dict[lang])
        result[lang] = reduced[idx:idx+n]
        idx += n
    
    return result

def plot_single_model(ax, reduced_embeddings, title, show_legend=True):
    """Plot embeddings for a single model on given axis."""
    
    for lang in ['EN', 'HI', 'CM']:
        coords = reduced_embeddings[lang]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=COLORS[lang],
            marker=MARKERS[lang],
            s=40,
            alpha=0.6,
            edgecolors='white',
            linewidths=0.5,
            label=lang
        )
    
    # ax.set_xlabel('Dimension 1')
    # ax.set_ylabel('Dimension 2')
    ax.set_title(title, fontweight='bold', pad=10)
    
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)
    
    ax.grid(True, alpha=0.3, linestyle='--')


def create_comparison_plot(model_reduced_list, model_names, layer_idx, method='t-SNE'):
    """Create comparison plot for multiple models."""
    
    n_models = len(model_names)
    n_cols = min(3, n_models)  # Max 3 columns
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Handle single plot case
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (reduced, model_name) in enumerate(zip(model_reduced_list, model_names)):
        display_name = NAME_MAP.get(model_name, model_name)
        plot_single_model(axes[idx], reduced, display_name, show_legend=False)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def create_multi_layer_plot(model_name, rep_dir, layers=[0, 4, 8, 12], n_samples=None):
    """Create a grid showing different layers for a single model."""
    
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    
    # Handle single layer case
    if n_layers == 1:
        axes = [axes]
    
    for col, layer_idx in enumerate(layers):
        print(f"  Processing Layer {layer_idx}...")
        
        # Load and reduce
        emb = load_representations(rep_dir, layer_idx, n_samples=n_samples)
        reduced = reduce_dimensions(emb, method='tsne', perplexity=30)
        plot_single_model(axes[col], reduced, f'Layer {layer_idx}', show_legend=False)
        
        # Clean up
        del emb, reduced
        gc.collect()
    
    # Set labels
    axes[0].set_ylabel('t-SNE Dimension 2', fontweight='bold', fontsize=12)
    
    for col in range(n_layers):
        axes[col].set_xlabel('t-SNE Dimension 1', fontsize=11)
        if col > 0:
            axes[col].set_ylabel('')
    
    # Shared legend
    legend_elements = [
        Patch(facecolor=COLORS['EN'], edgecolor='white', label='English (EN)'),
        Patch(facecolor=COLORS['HI'], edgecolor='white', label='Hindi (HI)'),
        Patch(facecolor=COLORS['CM'], edgecolor='white', label='Code-Mixed (CM)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
               bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=True)
    
    fig.suptitle(f'{model_name} - Layer-wise Representation Comparison (t-SNE)', 
                 fontsize=18, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    return fig


def create_density_plot(model_reduced_list, model_names, layer_idx):
    """Create density/contour plots showing cluster concentrations."""
    from scipy.stats import gaussian_kde
    
    n_models = len(model_names)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Handle single plot case
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (reduced, model_name) in enumerate(zip(model_reduced_list, model_names)):
        ax = axes[idx]
        display_name = NAME_MAP.get(model_name, model_name)
        
        for lang in ['EN', 'HI', 'CM']:
            coords = reduced[lang]
            
            # Scatter with density colors
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=COLORS[lang],
                marker=MARKERS[lang],
                s=50,
                alpha=0.5,
                edgecolors='white',
                linewidths=0.5,
                label=lang
            )
            
            # Add density contours
            try:
                xy = coords.T
                kernel = gaussian_kde(xy)
                xmin, xmax = coords[:, 0].min() - 1, coords[:, 0].max() + 1
                ymin, ymax = coords[:, 1].min() - 1, coords[:, 1].max() + 1
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                zz = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contour(xx, yy, zz, colors=COLORS[lang], alpha=0.6, levels=3, linewidths=1.5)
            except:
                pass
        
        # ax.set_xlabel('t-SNE Dimension 1')
        # ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(display_name, fontweight='bold', pad=10)
        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Sentence Representations with Density Contours (t-SNE, Layer {layer_idx})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def save_legend_strip(output_path):
    """Save a standalone legend strip as a separate file."""
    fig, ax = plt.subplots(figsize=(5, 0.5))
    ax.axis('off')
    legend_elements = [
        Patch(facecolor=COLORS['EN'], label='English (EN)'),
        Patch(facecolor=COLORS['HI'], label='Hindi (HI)'),
        Patch(facecolor=COLORS['CM'], label='Code-Mixed (CM)'),
    ]
    ax.legend(handles=legend_elements, loc='center', ncol=3,
              frameon=True, fancybox=True, fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved legend strip to {output_path}")


def process_model_group(model_list, group_name, base_output_dir, layer_idx, n_samples, n_samples_multilayer):
    """Process a group of models (BERT or RoBERTa family)."""
    
    print(f"\n{'='*80}")
    print(f"Processing {group_name} models...")
    print(f"{'='*80}")
    
    model_reduced_list = []
    model_names = []
    
    # Load and reduce all models
    for model_name in model_list:
        rep_dir = os.path.join(BASE_REP_DIR, model_name)
        
        # Check if directory exists
        if not os.path.exists(rep_dir):
            print(f"⚠ Warning: Directory not found for {model_name}, skipping...")
            continue
        
        print(f"\n{model_name}:")
        print(f"  Loading Layer {layer_idx} (sampling {n_samples} per language)...")
        
        try:
            emb = load_representations(rep_dir, layer_idx, n_samples=n_samples)
            print(f"  Reducing dimensions with t-SNE...")
            reduced = reduce_dimensions(emb, method='tsne', perplexity=30)
            
            model_reduced_list.append(reduced)
            model_names.append(model_name)
            
            del emb
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
            continue
    
    if not model_reduced_list:
        print(f"⚠ No models successfully processed for {group_name}")
        return
    
    # Create group output directory
    group_output_dir = os.path.join(base_output_dir, f"{group_name}_family")
    os.makedirs(group_output_dir, exist_ok=True)
    
    # Create comparison plot for all models in group
    print(f"\nGenerating {group_name} family comparison plot...")
    fig1 = create_comparison_plot(model_reduced_list, model_names, layer_idx)
    fig1.savefig(f"{group_output_dir}/representation_comparison_tsne.png", dpi=300, bbox_inches='tight')
    fig1.savefig(f"{group_output_dir}/representation_comparison_tsne.pdf", bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Saved to {group_output_dir}/representation_comparison_tsne.png")
    
    # Create density plot for all models in group
    print(f"Generating {group_name} family density plot...")
    fig2 = create_density_plot(model_reduced_list, model_names, layer_idx)
    fig2.savefig(f"{group_output_dir}/representation_density_tsne.png", dpi=300, bbox_inches='tight')
    fig2.savefig(f"{group_output_dir}/representation_density_tsne.pdf", bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved to {group_output_dir}/representation_density_tsne.png")
    
    # Clean up before individual model processing
    del model_reduced_list
    gc.collect()
    
    # Reset sample indices for multi-layer
    global SAMPLE_INDICES
    SAMPLE_INDICES = None
    
    # Process each model individually for multi-layer plots
    for model_name in model_names:
        print(f"\nGenerating multi-layer plot for {model_name}...")
        
        # Create model-specific output directory
        model_output_dir = os.path.join(base_output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        rep_dir = os.path.join(BASE_REP_DIR, model_name)
        
        # Reset sample indices for this model
        SAMPLE_INDICES = None
        
        try:
            fig3 = create_multi_layer_plot(model_name, rep_dir, layers=[0, 4, 8, 12], n_samples=n_samples_multilayer)
            fig3.savefig(f"{model_output_dir}/representation_multilayer_tsne.png", dpi=300, bbox_inches='tight')
            fig3.savefig(f"{model_output_dir}/representation_multilayer_tsne.pdf", bbox_inches='tight')
            plt.close(fig3)
            print(f"✓ Saved to {model_output_dir}/representation_multilayer_tsne.png")
            
            # Reset for next model
            SAMPLE_INDICES = None
            gc.collect()
            
        except Exception as e:
            print(f"✗ Error creating multi-layer plot for {model_name}: {e}")


def main():
    # Configuration
    N_SAMPLES = 500  # Sample size for single layer
    BASE_OUTPUT_DIR = f"/output_dir/TSNE/sampled_draft_aligned_{N_SAMPLES}"
    LAYER_IDX = 12  # Last layer
    N_SAMPLES_MULTILAYER = 500  # Smaller sample for multi-layer
    
    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Save standalone legend strip
    save_legend_strip(f"{BASE_OUTPUT_DIR}/legend_strip.png")
    
    # Process BERT family models
    process_model_group(BERT_MODELS, "BERT", BASE_OUTPUT_DIR, LAYER_IDX, N_SAMPLES, N_SAMPLES_MULTILAYER)
    
    # Process RoBERTa family models
    process_model_group(ROBERTA_MODELS, "RoBERTa", BASE_OUTPUT_DIR, LAYER_IDX, N_SAMPLES, N_SAMPLES_MULTILAYER)
    
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()