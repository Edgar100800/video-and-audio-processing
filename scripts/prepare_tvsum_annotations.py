import h5py
import numpy as np
from pathlib import Path

# TVSum .mat is v7.3 format (HDF5)
data = h5py.File('data/raw/tvsum/annotations/ydata-tvsum50.mat', 'r')

# The structure is different from scipy .mat
# videos are referenced via HDF5 references
video_ids = data['tvsum50']['video']
features_dir = Path('data/features')

for i in range(len(video_ids)):
    # Get video ID (it's stored as bytes)
    vid_ref = video_ids[i]
    vid_data = data[vid_ref]
    video_id = ''.join(chr(c) for c in vid_data[:].flatten())
    
    # Get scores
    scores_ref = data['tvsum50']['user_anno'][i]
    scores_data = data[scores_ref][:]
    
    # Average across users (20 annotators)
    shot_scores = np.mean(scores_data, axis=0)
    
    h5_path = features_dir / f"{video_id}.h5"
    if h5_path.exists():
        with h5py.File(h5_path, 'a') as f:
            n_frames = f['n_frames'][()]
            # TVSum shots are ~2 seconds, at 2 FPS that's ~4 frames per shot
            # Simple interpolation: repeat each shot score
            frame_scores = np.repeat(shot_scores, 4)[:n_frames]
            frame_scores = frame_scores.astype(np.float32)
            
            if 'gt_scores' in f:
                del f['gt_scores']
            f.create_dataset('gt_scores', data=frame_scores)
        print(f"Updated {video_id}: {n_frames} frames, scores {frame_scores.min():.2f}-{frame_scores.max():.2f}")
    else:
        print(f"Warning: {h5_path} not found")

data.close()
print("Annotations prepared!")
