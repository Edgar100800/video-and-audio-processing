import h5py
import numpy as np
from pathlib import Path

# TVSum .mat is v7.3 format (HDF5)
data = h5py.File('data/raw/tvsum/annotations/ydata-tvsum50.mat', 'r')

# The structure uses HDF5 references
# Access the tvsum50 group
tvsum50 = data['tvsum50']

# video IDs are stored as cell array with HDF5 references
video_ids = tvsum50['video']
user_anno = tvsum50['user_anno']

features_dir = Path('data/features')

for i in range(len(video_ids)):
    # Dereference the HDF5 reference to get the actual dataset
    vid_ref = video_ids[i][0]  # Get reference from cell array
    vid_dataset = data[vid_ref]
    
    # Read video ID - stored as uint16 characters
    video_id_chars = vid_dataset[:].flatten()
    video_id = ''.join(chr(int(c)) for c in video_id_chars)
    
    # Get user annotations
    anno_ref = user_anno[i][0]
    anno_dataset = data[anno_ref]
    scores_data = anno_dataset[:]
    
    # Average across users (20 annotators)
    shot_scores = np.mean(scores_data, axis=0)
    
    h5_path = features_dir / f"{video_id}.h5"
    if h5_path.exists():
        with h5py.File(h5_path, 'a') as f:
            n_frames = f['n_frames'][()]
            # TVSum shots are ~2 seconds, at 2 FPS that's ~4 frames per shot
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
