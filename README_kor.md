# USGS 하천유량 데이터로더 (USGS Streamflow Dataloaders)

이 저장소는 **USGS 하천유량(streamflow) 데이터**를 읽어들이고,
[Torch Spatiotemporal Library (TSL)](https://github.com/scdmlab/tsl)로 학습하기 위한
그래프 기반 입력 텐서를 준비하는 두 개의 커스텀 데이터로더를 제공합니다.

---

## 🚀 개요

이 데이터로더들은 TSL 기반 모델(예: HydroTGNN 또는 수문학용 그래프 신경망 아키텍처)과
매끄럽게 통합되도록 설계되었습니다.
USGS 관측소(gauge) 데이터셋으로 시공간(spatiotemporal) 모델을 학습하는 데 필요한
전처리, 시간축 분할(temporal slicing), 그래프 구성 단계를 처리합니다.

> 🔹 데이터셋이나 예제 분할(split)에 접근이 필요하면 아래로 문의하세요.
> **📧 tsui5@wisc.edu**

---

## 📁 저장소 구성

| 파일 | 설명 |
|------|--------------|
| `hydrology_dataset.py` | USGS 하천유량 데이터를 읽고 전처리하여 TSL 모델과 호환되는 시간 그래프 텐서를 생성하는 커스텀 데이터셋 클래스. |
| `spin_hydrology.yaml` | 수문 데이터셋으로 **임의의 TSL 모델**을 학습하기 위한 범용 설정 파일. 데이터 경로, 그래프 연결성(connectivity), 데이터 분할, 학습 하이퍼파라미터를 정의합니다. |
| *(외부)* [`tsl/`](https://github.com/scdmlab/tsl) | 전체 시공간 모델링 프레임워크 (재현성과 이 데이터로더들과의 통합을 위해 fork된 버전). |

---

## 🔧 사용 예시

1. **저장소 클론**
   ```bash
   git clone https://github.com/your-username/USGS-Streamflow-Dataloaders.git
   git clone https://github.com/scdmlab/tsl.git
