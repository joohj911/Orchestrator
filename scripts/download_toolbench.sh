#!/usr/bin/env bash
# ============================================================
# ToolBench 데이터 다운로드 (공식 OpenBMB/ToolBench data.zip).
#
# 사용법:
#   bash scripts/download_toolbench.sh [DEST]
#   DEST 기본값 = ./data/toolbench  (config.yaml 의 paths.toolbench_root 와 일치시킬 것)
#
# 결과: DEST/data/test_instruction/G{1,2,3}_instruction.json 이 생성된다.
#   m1_data_prep.py 는 {toolbench_root}/data/test_instruction/... 를 자동 탐색하므로,
#   config.paths.toolbench_root = DEST 로 두면 그대로 인식된다.
#
# 주의:
#   - data.zip 은 대용량(수 GB, toolenv 포함)이다. M1 은 test_instruction 만 쓰지만
#     공식 배포가 단일 zip 이라 전체를 받는다. 디스크 여유 확인.
#   - Google Drive 대용량 파일은 confirm 토큰이 필요해 wget 대신 gdown 을 쓴다.
#   - Drive 할당량 초과(quota exceeded)로 실패하면 아래 Tsinghua Cloud 를 사용:
#       https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/
#     (수동 다운로드 후 DEST 에서 `unzip data.zip`)
#   - 근거: OpenBMB/ToolBench README (data.zip file id, 압축 구조).
# ============================================================
set -euo pipefail

DEST="${1:-./data/toolbench}"
FILEID="1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"   # 공식 data.zip (ToolBench README)

mkdir -p "$DEST"
cd "$DEST"

if [ -d "$DEST/data/test_instruction" ]; then
  echo "[skip] 이미 존재: $DEST/data/test_instruction"
  ls -1 "$DEST/data/test_instruction" | head
  exit 0
fi

echo "[1/3] gdown 설치/업데이트"
python -m pip install -q --upgrade gdown

echo "[2/3] data.zip 다운로드 (Google Drive, 대용량 — 시간 소요)"
if ! gdown "https://drive.google.com/uc?id=${FILEID}" -O data.zip; then
  echo "!!! Google Drive 다운로드 실패 (할당량 초과 가능)." >&2
  echo "!!! Tsinghua Cloud 에서 수동 다운로드 후 '$DEST' 에 data.zip 을 두고" >&2
  echo "!!!   cd '$DEST' && unzip data.zip   를 실행하세요:" >&2
  echo "!!!   https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/" >&2
  exit 1
fi

echo "[3/3] 압축 해제"
unzip -q -o data.zip

if [ ! -d "$DEST/data/test_instruction" ]; then
  echo "!!! 압축 해제 후에도 $DEST/data/test_instruction 이 없습니다." >&2
  echo "!!! zip 내부 구조를 확인하세요: unzip -l '$DEST/data.zip' | head" >&2
  exit 1
fi

echo "완료. test_instruction 파일:"
ls -1 "$DEST/data/test_instruction"
echo ""
echo "다음: config.yaml 의 paths.toolbench_root='$DEST' 확인 후"
echo "  python src/m1_data_prep.py --config config.yaml"
