from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lab_utils.visualization import plot_class_balance, plot_numeric_distribution
SEED = 1234
SPLITS = ('train', 'val', 'test')
LABELS = ('cat', 'dog')
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')

def list_image_paths_for_group(data_root: Path, split: str, label: str) -> list[Path]:
    path = data_root / split / label
    return sorted((p for pat in IMAGE_EXTENSIONS for p in path.glob(pat)))
    raise NotImplementedError('List the image paths for one split and one label.')

def inspect_image_file(path: Path) -> tuple[int, int, float]:
    pil_image = Image.open(path).convert('RGB')
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    mean_intensity = np_image.mean()
    return (pil_image.width, pil_image.height, mean_intensity)
    raise NotImplementedError('Inspect one image file.')

def make_metadata_row(path: Path, data_root: Path, split: str, label: str) -> dict[str, object]:
    width, height, mean_intensity = inspect_image_file(path)
    return {'filepath': str(path.relative_to(data_root)), 'label': label, 'split': split, 'width': width, 'height': height, 'mean_intensity': mean_intensity}
    raise NotImplementedError('Create one metadata row from one image path.')

def build_metadata_from_folders(data_root: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        for label in LABELS:
            paths = list_image_paths_for_group(data_root, split, label)
            rows.extend((make_metadata_row(p, data_root, split, label) for p in paths))
    return pd.DataFrame(rows).sort_values(['split', 'label', 'filepath']).reset_index(drop=True)

def load_metadata_table(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    return data
    raise NotImplementedError('Load the saved metadata table with Pandas.')

def summarize_metadata(frame: pd.DataFrame) -> dict[str, object]:
    return {'rows': len(frame), 'columns': list(frame.columns), 'class_counts': frame['label'].value_counts(), 'split_counts': frame['split'].value_counts()}
    raise NotImplementedError('Summarize the metadata table with Pandas.')

def build_label_split_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(['label', 'split']).size().unstack(fill_value=0)
    raise NotImplementedError('Build the label-by-split count table.')

def audit_metadata(frame: pd.DataFrame) -> dict[str, object]:
    duplicate_filepath = frame['filepath'].duplicated().sum()
    missing_filepaths = frame['filepath'].isna().sum()
    missing_label = frame['label'].isna().sum()
    missing_splits = frame['split'].isna().sum()
    missing_widths = frame['width'].isna().sum()
    missing_heights = frame['height'].isna().sum()
    missing_mean_intensity = frame['mean_intensity'].isna().sum()
    invalid_sizes = 0
    bad_labels = []
    for label in frame['label'].unique():
        if label != 'cat' and label != 'dog':
            bad_labels.append(label)
    for width, height in zip(frame['width'], frame['height']):
        if width <= 0 or height <= 0:
            invalid_sizes += 1
    return {'missing_values': {'filepath': missing_filepaths, 'label': missing_label, 'split': missing_splits, 'width': missing_widths, 'height': missing_heights, 'mean_intensity': missing_mean_intensity}, 'duplicate_filepaths': duplicate_filepath, 'bad_labels': bad_labels, 'non_positive_sizes': invalid_sizes}
    raise NotImplementedError('Audit the metadata table.')

def add_analysis_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df_copy = frame.copy()
    brightness_band = pd.qcut(df_copy['mean_intensity'], q=4, labels=['darkest', 'dim', 'bright', 'brightest'])
    df_copy['pixel_count'] = df_copy['width'] * df_copy['height']
    df_copy['aspect_ratio'] = df_copy['width'] / df_copy['height']
    df_copy['brightness_band'] = brightness_band
    ref = 64 * 64
    df_copy['size_bucket'] = df_copy['pixel_count'].apply(lambda pc: 'small' if pc < ref else 'large' if pc > ref else 'medium')
    return df_copy
    raise NotImplementedError('Create the analysis columns with Pandas.')

def build_split_characteristics_table(frame: pd.DataFrame) -> pd.DataFrame:
    avg_width = frame.groupby('split')['width'].mean()
    avg_height = frame.groupby('split')['height'].mean()
    avg_intensity = frame.groupby('split')['mean_intensity'].mean()
    avg_pixel_count = frame.groupby('split')['pixel_count'].mean()
    return pd.DataFrame({'avg_width': avg_width, 'avg_height': avg_height, 'avg_mean_intensity': avg_intensity, 'avg_pixel_count': avg_pixel_count})
    raise NotImplementedError('Build the split characteristics summary table.')

def sample_balanced_by_split_and_label(frame: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    sampled_dfs = []
    for _, group in frame.groupby(['split', 'label']):
        sampled_group = group.sample(n=min(n_per_group, len(group)), random_state=seed)
        sampled_dfs.append(sampled_group)
    return pd.concat(sampled_dfs, ignore_index=True)
    raise NotImplementedError('Build a balanced sample across split and label groups.')
sample_size_per_group = 5
