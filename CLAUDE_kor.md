# CLAUDE.md (한국어판)

이 파일은 이 저장소에서 작업할 때 Claude Code(claude.ai/code)에 제공되는 안내입니다.
영문 원본은 `CLAUDE.md`이며, 이 문서는 그 번역본입니다.

## 이 저장소는 무엇인가

Torch Spatiotemporal 라이브러리(tsl) 위에 얹은 얇은 수문학(hydrology) 레이어입니다.
프로젝트 파일은 최상위에 있고, 나머지는 전부 tsl 프레임워크입니다.

- `hydrology_dataset.py` — `IJCNNHydrologyDataset(TabularDataset)`: USGS 관측소
  인접행렬 CSV와 유량(discharge) CSV를 읽어 tsl의 데이터셋 API에 맞게 변환합니다.
- `spin_hydrology.yaml` — 위 데이터셋으로 SPIN 결측보간(imputation) 모델을 학습하기 위한
  Hydra 설정. `default`(즉 `tsl/examples/imputation/config/default.yaml`) 위에 합성(compose)되므로,
  해당 config 디렉터리 안에 두거나 그쪽을 가리키도록 해야 합니다.
- `tutorial.ipynb` — 합성 8개 관측소 하천망을 이용한 엔드투엔드 워크스루(한국어):
  CSV → 데이터셋 → 연결성(connectivity) → 마스크 → `ImputationDataset` → DataModule →
  `Imputer`/GRIN → 평가, 여기에 예측(forecasting) 변형과 Hydra 설정 연결 방법까지 포함합니다.
  스크립트로부터 생성되었고 오류 없이 실행되었습니다. 이 노트북의 `HydrologyDataset` 서브클래스가
  아래 "데이터셋 함정"에 나열된 수정 사항들의 기준 구현입니다.
  저장소 루트에서 `hydrotgnn` 커널로 실행하세요.
- `tsl/` — https://github.com/scdmlab/tsl (TorchSpatiotemporal/tsl의 fork, v0.9.6)의 전체 클론입니다.
  **중첩된 git 저장소이며 서브모듈이 아니고, 바깥 저장소에서는 추적되지 않습니다(untracked).**
  `tsl/` 안에서 `git status`를 보면 약 278개 파일이 수정된 것으로 보이지만,
  "tsl 로컬 패치"에 나열된 세 파일을 제외하면 전부 CRLF/LF 줄바꿈 차이일 뿐입니다.
  이를 "고치거나" 커밋하지 마세요.

원시 데이터(`IJCNN/adj_matrix.csv`, `IJCNN/merged_discharge_data.csv`)는 **저장소에 없습니다**.
README는 tsui5@wisc.edu 로 이메일을 보내 요청하라고 안내합니다. `spin_hydrology.yaml`은 두 파일 모두
Windows 절대경로(`E:/WISC/...`)로 하드코딩되어 있으므로 Linux에서는 반드시 덮어써야 합니다.
`tutorial.ipynb`는 같은 형식의 합성 CSV를 `data/synthetic/`(gitignore 대상)에 씁니다.

## 환경

**항상 conda 환경 `hydrotgnn`을 사용하세요** (`/home/hydro/miniconda3/envs/hydrotgnn`, Python 3.11).
명령 앞에 `conda run -n hydrotgnn`을 붙이거나 해당 환경의 `bin/python` / `bin/pip`를 직접 호출하세요.
설치 및 검증 완료: torch 2.11+cu128 (RTX 5090에는 cu128 필요), torch_geometric 2.8,
torch_scatter / torch_sparse (`https://data.pyg.org/whl/torch-2.11.0+cu128.html` 에서 설치),
lightning 2.6, hydra-core, jupyter, pytest, 그리고 editable 설치된 tsl.
`hydrotgnn`이라는 이름의 Jupyter 커널이 등록되어 있습니다.

**tsl은 반드시 `--config-settings editable_mode=compat`로 설치해야 합니다.** 저장소 루트에서는
그렇지 않을 경우 `import tsl`이 `tsl/` 클론 디렉터리를 네임스페이스 패키지로 해석해 버립니다
(`__version__` 없음, 서브모듈 없음). 기본 editable finder가 경로 스캔 이후에 실행되기 때문입니다.
그런 상황이 되면 다시 설치하세요:

```bash
/home/hydro/miniconda3/envs/hydrotgnn/bin/pip install --no-deps --config-settings editable_mode=compat -e tsl
```

## 명령어

모든 tsl 명령은 `tsl/`에서 실행합니다.

```bash
# 테스트 (pytest.ini에 마커 정의: slow, integration). 빠른 스위트: 44개 통과, 약 4초.
cd tsl && python -m pytest -q -m 'not slow'
python -m pytest tests/test_metrics.py::test_name -v     # 단일 테스트
python -m pytest tests/test_example_imputation.py -v    # 1배치 imputation 스모크 테스트 (slow, integration)

# 린트/포맷 (tsl의 pre-commit: isort, yapf, flake8 --max-line-length=80)
cd tsl && pre-commit run --all-files

# 실험 실행 (Hydra). `config=`와 `config_path=`는 tsl에서 --config-name / --config-path의
# 축약형이며, 그 외 key=value는 합성된 설정을 덮어씁니다.
cd tsl/examples/imputation && python run_imputation_experiment.py config=spin dataset.name=la
cd tsl/examples/forecasting && python run_traffic_experiment.py model=dcrnn dataset=la

# 튜토리얼 노트북 (저장소 루트에서). 출력을 헤드리스로 재생성하려면:
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=hydrotgnn --inplace tutorial.ipynb

# 수문 데이터셋만 단독 스모크 테스트 (cwd 기준 ./IJCNN/*.csv 필요)
python hydrology_dataset.py
```

실험 출력물은 cwd 기준 `logs/imputation/<model>/<date>/<time>/`에 저장됩니다
(`default.yaml`의 `hydra.run.dir`): TensorBoard 로그, 최적 체크포인트, 해석된(resolved) 설정.
tsl은 벤치마크 데이터셋을 기본적으로 `tsl/tsl/.storage`에 내려받습니다
(`tsl.config.data_dir`, 설정의 `data_dir=`로 변경 가능). `logs/`와 `data/`는 gitignore 대상입니다.

## tsl 로컬 패치 (PyG 2.8 / torch 2.6+ 대응)

tsl 0.9.6은 이 라이브러리 버전들보다 먼저 나왔습니다. `tsl/tsl/` 아래 세 파일에 주석이 달린
작은 호환성 패치가 들어 있습니다. `tsl/`을 다시 클론하거나 업데이트하더라도 이 패치는 유지하세요.

- `transforms/imputation.py`, `transforms/rearrange.py`, `transforms/masked_subgraph.py` —
  PyG ≥2.7부터 `BaseTransform.forward`가 추상 메서드가 되었고 `__call__`이 `Data`를 얕은 복사(shallow copy)합니다.
  이 얕은 복사가 tsl의 `Data.input` / `Data.target` 뷰를 깨뜨립니다(뷰가 예전 스토리지를 계속 가리켜서,
  `.to('cuda')` 이후에도 뷰 텐서가 CPU에 남아 "Expected all tensors to be on the same device" 오류 발생).
  이제 각 transform은 `forward`를 구현하고 `__call__`을 오버라이드해 in-place로 호출합니다.
- `engines/predictor.py`의 `load_model` — `torch.load`에 `weights_only=False`를 전달합니다
  (torch ≥2.6의 기본값 `True`는 체크포인트에 피클된 tsl 메트릭/모델 클래스 객체를 거부합니다).

## 아키텍처: 실행이 tsl을 통과하는 흐름

```
CSV 파일들 --> HydrologyDataset (DatetimeDataset; pandas, 컬럼 = MultiIndex(nodes, channels))
          --> ImputationDataset (torch; `window`/`stride`의 슬라이딩 윈도우, 연결성, 마스크)
          --> SpatioTemporalDataModule (타깃에 StandardScaler, val/test 시간 분할기)
          --> Imputer (모델 클래스를 감싸는 LightningModule, loss = MaskedMAE)
          --> pytorch_lightning.Trainer (val_mae 기준 EarlyStopping + ModelCheckpoint)
```

`tsl.experiment.Experiment`는 `hydra.main`을 감쌉니다. 시드를 설정하고 `cfg.run.dir`를 지정한 뒤
`run_fn`을 호출합니다. 모델은 각 예제 스크립트의 `get_model_class`에서 문자열로 선택하고
(imputation의 경우 `rnni`, `birnni`, `grin`, `spin`, `spin-h`), 데이터셋은 `get_dataset`에서 문자열로 선택합니다.

### 수문 데이터 연결(wiring)은 아직 미완성

`spin_hydrology.yaml`은 `dataset.name: hydrology`와 `adj_matrix_path` / `discharge_data_path`를
설정하지만, `run_imputation_experiment.py`의 `get_dataset`은 `air*`, `la`, `bay`만 알고 있으며
그 외에는 `ValueError`를 던집니다. 또한 `tsl/` 안의 어떤 코드도 `hydrology_dataset.py`를
import하지 않습니다. 추가해야 할 `get_dataset` 분기는 `tutorial.ipynb`의 11번 섹션에 나와 있습니다.
데이터셋 자체도 러너가 동작하려면 아래 수정 사항들이 먼저 필요합니다.

### `IJCNNHydrologyDataset`의 데이터셋 함정 (모두 노트북의 `HydrologyDataset`에서 수정됨)

- **마스크 극성이 반대입니다.** tsl의 `mask`는 *관측됨(observed)* 위치가 True인데,
  `create_mask()`는 `df.isna()`(결측 위치가 True)를 반환하고 `self.mask`에 DataFrame을 저장합니다.
  대신 베이스 생성자에 `mask=~df.isna()`를 전달하세요.
- **엣지 방향이 뒤집혀 있습니다.** tsl/PyG는 `A[i, j]`를 엣지 *j → i*로 읽습니다
  (`adj_to_edge_index`가 전치함). 반면 CSV에서 `a[i, j] > 0`은 *i → j*(상류 → 하류)를 의미합니다.
  `prepare_distance_matrix()`가 전치를 하지 않아 메시지가 하류 → 상류로 흐릅니다. `dist.T`를 저장하세요.
- **`compute_similarity`가 구현되어 있지 않고** `similarity_options`도 설정되지 않아
  `get_connectivity(method='distance')`가 예외를 던집니다. `gaussian_kernel(self.dist, theta)`를 쓰세요.
  이진 인접행렬(거리 0/1/inf)에서는 MetrLA 방식의 `theta = std(유한 거리들)` ≈ 0.5가 되어
  가중치가 ≈ 0.02가 되는데, yaml의 `threshold: 0.1`이 이를 전부 잘라내어
  **엣지가 하나도 없는 빈 그래프가 되고 오류도 나지 않습니다**.
  `theta=1.0`(가중치 ≈ 0.37)을 사용하고 엣지 개수를 assert로 확인하세요.
- **`datetime_encoded`는 `TabularDataset`이 아니라 `DatetimeDataset`에 있습니다.**
  (모든 내장 tsl 데이터셋이 그렇듯) `DatetimeDataset`을 상속해야 러너의 day-of-time 공변량이 동작합니다.
- **`eval_mask` / `training_mask`**는 (러너가 `la`/`bay`에 대해 하듯)
  `tsl.ops.imputation.add_missing_values`로 감쌀 때 생성됩니다.
- **GRIN은 노드 단위 exog가 필요합니다.** `GRINModel`은 `u`를 노드별 텐서와 concat하므로
  전역 `(T, 2)` 공변량(예제 스크립트가 전달하는 형태)은 "Tensors must have same number of dimensions"로
  실패합니다. `(T, N, 2)`로 브로드캐스트하세요. 또한 GRIN은 `merge_mode='mlp'`일 때 `embedding_size`가 필요합니다.
- 인접행렬 CSV와 유량 CSV의 노드 수가 다르면, 데이터셋은 조용히 둘 다 앞쪽 `min(n)`개 노드로 잘라냅니다.
  정렬이 맞다고 가정하지 말고 출력되는 경고를 확인하세요.

## 기타 참고사항

- `README.md`는 앞부분에 불필요한 `readme = """`로 시작하며 "Usage Example" 중간에서 잘려 있습니다.
  git status에 보이는 `M README.md`는 CRLF→LF 변경일 뿐입니다.
- `spin_hydrology.yaml`의 중국어 주석은 각각 "절대 경로"와
  "수문 데이터에 맞게 조정된 파라미터"(`epochs: 10`, `batch_size: 4`)를 뜻합니다.
