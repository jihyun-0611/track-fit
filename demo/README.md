# Track-Fit 데모

실시간 웹캠 프레임에서 MediaPipe keypoint를 추출하고, 학습 파이프라인과 동일한
COCO20 입력으로 변환한 뒤 calibrated ProtoGCN ensemble로 운동 클래스를 예측한다.

## Ensemble 구성

실시간 앱은 `demo/configs/ensemble.yaml`에서 모든 stream 설정을 읽는다. 현재 구성은
`protogcn/tools/run_ensemble.py`의 validation temperature 보정과 internal split weight
탐색 결과를 반영한다.

| Stream | 입력 feature | Config | Weight | Temperature |
| --- | --- | --- | ---: | ---: |
| `pt05` | joint (`j`) | `demo/configs/streams/j.yaml` | 0.50 | 0.8 |
| `b_T2` | bone (`b`) | `demo/configs/streams/b.yaml` | 0.25 | 0.7 |
| `a_basic` | joint angle (`a`) | `demo/configs/streams/a.yaml` | 0.25 | 0.9 |

각 stream의 softmax 확률은 해당 temperature로 보정된 후 weight에 따라 가중 평균된다.

## 환경 설정

프로젝트 루트의 `.env`에 checkpoint 루트 경로를 설정해야 한다.

```dotenv
WORK_DIR=/path/to/track-fit/work_dirs
```

`ensemble.yaml`의 checkpoint 경로는 `WORK_DIR`을 기준으로 해석된다. 현재 지정된
checkpoint는 다음과 같다.

- `protect_torso_c05/best_top1_acc_epoch_120.pth`
- `b_T2/best_top1_acc_epoch_100.pth`
- `a_basic/best_top1_acc_epoch_60.pth`

## 입력 변환

- MediaPipe 33개 landmark에서 학습과 동일한 COCO17 관절을 선택한다.
- 정규화 좌표를 원본 프레임의 pixel 좌표로 변환한다.
- mid-hip, spine, mid-shoulder pseudo joint를 confidence까지 포함해 평균한다.
- pose 미검출 프레임은 `(20, 3)` zero 배열로 시간축에 유지한다.
- 최근 추론 윈도우의 미검출 프레임 비율이 50%를 초과하면 예측하지 않는다.

## 실행

프로젝트의 `mediapipe` Conda 환경을 사용한다.

```bash
conda activate mediapipe
cd demo
bash run_demo.sh
```

서버가 시작되면 `http://localhost:8000`에서 접속할 수 있다.
