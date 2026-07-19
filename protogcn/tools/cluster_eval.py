import argparse, pickle
import numpy as np

CLUSTERS = {
    'C1_bench_family': ['bench_press', 'incline_bench_press', 'decline_bench_press'],
    'C2_plank_bench':  ['plank', 'bench_press'],
    'C3_press_pull':   ['shoulder_press', 'lat_pulldown', 'pull_up'],
    'C4_hinge':        ['deadlift', 'romanian_deadlift'],
}
# 7모델 공통 오답 9샘플 (frame_dir 기준 추적)
STRUCTURAL = ['bench_press_2', 'hammer_curl_2', 'hammer_curl_3',
              'incline_bench_press_2', 'incline_bench_press_3',
              'lateral_raise_1', 'plank_2', 'shoulder_press_2', 't_bar_row_3']

def main(args):
    d = pickle.load(open(args.dataset_pkl, 'rb'))
    fds = d['split'][args.split]
    ann = {a['frame_dir']: a for a in d['annotations']}
    labels = np.array([ann[f]['label'] for f in fds])
    exercises = [ann[f]['exercise'] for f in fds]
    scores = np.array(pickle.load(open(args.pred, 'rb')))
    assert len(scores) == len(labels), f"{len(scores)} vs {len(labels)}"
    pred = scores.argmax(1)
    correct = pred == labels

    print(f"overall: {correct.sum()}/{len(labels)} = {correct.mean():.4f}")
    for cname, exs in CLUSTERS.items():
        idx = [i for i, e in enumerate(exercises) if e in exs]
        print(f"{cname}: {correct[idx].sum()}/{len(idx)}")
    hit = [f for i, f in enumerate(fds) if correct[i]
           and any(s in f for s in STRUCTURAL)]
    print(f"structural_9 recovered: {len(hit)} -> {hit}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pred', required=True)
    p.add_argument('--dataset-pkl', required=True)
    p.add_argument('--split', default='final_test')
    main(p.parse_args())