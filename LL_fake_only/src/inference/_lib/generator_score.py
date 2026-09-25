import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _generator_from_path(path):
    """Extract second-to-last directory component as the generator name."""
    parts = path.replace('\\', '/').split('/')
    return parts[-2] if len(parts) >= 2 else 'unknown'


def assign_generator_index(df, label_col='label'):
    """Add 'generator' and 'index' columns to df.

    Fakes get their parent directory as generator name; reals get 'real'.
    Sorted fake generators are indexed 0..N-1, real gets N.
    """
    df = df.copy()
    df.loc[df[label_col] == 'fake', 'generator'] = (
        df.loc[df[label_col] == 'fake', 'path'].apply(lambda x: x.split('/')[-2])
    )
    df.loc[df[label_col] == 'real', 'generator'] = 'real'

    generator_names = sorted(df[df[label_col] == 'fake']['generator'].unique())
    gen_to_idx = {g: i for i, g in enumerate(generator_names)}
    gen_to_idx['real'] = len(generator_names)
    df['index'] = df['generator'].map(gen_to_idx)
    return df, generator_names, gen_to_idx


def compute_per_generator_scores(df, gen_to_idx, generator_names,
                                  prediction_col='prediction'):
    """Per-generator real/fake prediction rates, broadcast back to per-row columns.

    Returns the augmented df (with 'real_score' and 'fake_score' columns)
    and the (real_score_map, fake_score_map) dicts.
    """
    real_score_map = {}
    fake_score_map = {}
    for gen in generator_names + ['real']:
        subset = df[df['generator'] == gen]
        real_score_map[gen_to_idx[gen]] = (subset[prediction_col] == 'real').mean()
        fake_score_map[gen_to_idx[gen]] = (subset[prediction_col] == 'fake').mean()

    df = df.copy()
    df['real_score'] = df['index'].map(real_score_map)
    df['fake_score'] = df['index'].map(fake_score_map)
    return df, real_score_map, fake_score_map


def save_generator_scores(df, csv_path='generator_score.csv', npy_path='generator_score.npy'):
    """Save the per-row (index, real_score, fake_score) view."""
    final_df = df[['index', 'real_score', 'fake_score']]
    final_df.to_csv(csv_path, index=False)
    np.save(npy_path, final_df.to_numpy())
    logger.info(f"Generator scores saved → {csv_path}, {npy_path}")
    return final_df


def save_generator_breakdown(results, csv_path='generator_breakdown.csv',
                              npy_path='generator_breakdown.npy'):
    """Aggregate per-generator stats (n, real_rate, fake_rate, avg_real_score).

    Used by CCL inference where we want a one-row-per-generator summary
    rather than the per-audio broadcast view.
    """
    if 'labels' not in results:
        logger.warning("No labels — skipping generator breakdown.")
        return None

    paths = results['paths']
    labels_bin = results['labels']
    predictions = results['predictions']
    prob_real = results['probabilities'][:, 1]

    generators = np.array([
        'real' if l == 1 else _generator_from_path(p)
        for p, l in zip(paths, labels_bin)
    ])

    rows = []
    for gen in sorted(set(generators)):
        mask = generators == gen
        rows.append({
            'generator':       gen,
            'n_samples':       int(mask.sum()),
            'true_label':      'real' if gen == 'real' else 'fake',
            'real_rate':       round(float((predictions[mask] == 1).mean()), 4),
            'fake_rate':       round(float((predictions[mask] == 0).mean()), 4),
            'avg_real_score':  round(float(prob_real[mask].mean()), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    np.save(npy_path, df.to_numpy())
    logger.info(f"Generator breakdown saved → {csv_path}, {npy_path}")
    return df
