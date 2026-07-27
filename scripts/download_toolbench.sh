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
# 배포 형태 (OpenBMB/ToolBench README):
#   - 단일 data.zip. 압축 해제 시 최상위에 data/ 가 생기고 그 안에 test_instruction/ 등.
#   - Google Drive 폴더: 1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL (안에 data.zip)
#   - data.zip 직접 file id: 1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk
#   - Tsinghua Cloud: https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/
#
# gdown 실패 대응 (gdown FAQ):
#   - 최신 gdown 이 대용량 파일의 바이러스검사 확인 페이지를 자동 처리 → 반드시 upgrade.
#   - 할당량 초과 시 gdown 이 HTML 에러 페이지를 data.zip 으로 저장할 수 있어, 받은 뒤
#     zip 유효성(unzip -l)을 반드시 검증한다.
#   - 그래도 안 되면 쿠키를 ~/.cache/gdown/cookies.txt 에 저장(브라우저 확장으로 export)
#     후 재시도, 또는 Tsinghua Cloud 수동 다운로드.
#   - data.zip 은 대용량(수 GB, toolenv 포함). 디스크 여유 확인.
# ============================================================
set -euo pipefail

DEST="${1:-./data/toolbench}"
FOLDER_ID="1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL"
FILE_ID="1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
TSINGHUA="https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/"

mkdir -p "$DEST"
cd "$DEST"

if [ -d "test_instruction" ] || [ -d "data/test_instruction" ]; then
  echo "[skip] 이미 존재: $DEST 아래 test_instruction"
  exit 0
fi

echo "[1/4] gdown 최신 버전 설치 (대용량 파일 confirm 자동 처리에 필요)"
python -m pip install -q --upgrade gdown

# zip 유효성 검사: 할당량 초과 시 gdown 이 HTML 을 저장하므로 실제 zip 인지 확인.
valid_zip() { [ -f "$1" ] && unzip -l "$1" >/dev/null 2>&1; }

echo "[2/4] data.zip 다운로드 시도 (에러는 그대로 출력하여 원인 진단)"
ZIP=""

# 전략 D(우선): Tsinghua Cloud 직접 다운로드 (Seafile ?dl=1). Google Drive 를 타지 않음.
#   Google Drive 할당량/권한/차단과 무관하므로 가장 안정적. 먼저 시도.
echo "  - 전략 D: Tsinghua Cloud 직접 다운로드"
rm -f data.zip
if command -v curl >/dev/null 2>&1; then
  curl -fL "${TSINGHUA}?dl=1" -o data.zip || true
elif command -v wget >/dev/null 2>&1; then
  wget -O data.zip "${TSINGHUA}?dl=1" || true
fi
if valid_zip data.zip; then ZIP="data.zip"; else echo "    (Tsinghua 실패 또는 유효하지 않은 zip)"; fi

# 전략 A: data.zip 직접 file id (Google Drive).
if [ -z "$ZIP" ]; then
  echo "  - 전략 A: Google Drive 직접 file id"
  rm -f data.zip
  gdown "$FILE_ID" -O data.zip --continue || true
  if valid_zip data.zip; then ZIP="data.zip"; else echo "    (실패)"; fi
fi

# 전략 C: 폴더 다운로드 후 내부 data.zip 탐색 (Google Drive).
#   (gdown 6.1.0 은 --fuzzy/--remaining-ok 미지원이라 해당 전략은 제거함.)
if [ -z "$ZIP" ]; then
  echo "  - 전략 C: Google Drive 폴더 통째 다운로드 후 data.zip 탐색"
  rm -rf gd_folder
  gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O gd_folder || true
  found="$(find gd_folder -name 'data.zip' 2>/dev/null | head -1)"
  if [ -n "$found" ] && valid_zip "$found"; then ZIP="$found"; else echo "    (실패)"; fi
fi

if [ -z "$ZIP" ]; then
  echo "" >&2
  echo "!!! 자동 다운로드 실패 (Google Drive 할당량 초과/권한 가능)." >&2
  echo "!!! 대안 1) 쿠키 설정 후 재시도:" >&2
  echo "!!!   브라우저에서 drive.google.com 쿠키를 ~/.cache/gdown/cookies.txt 로 export 후" >&2
  echo "!!!   본 스크립트 재실행 (gdown 이 자동으로 쿠키 사용)." >&2
  echo "!!! 대안 2) Tsinghua Cloud 에서 수동 다운로드 후 압축 해제:" >&2
  echo "!!!   $TSINGHUA" >&2
  echo "!!!   다운받은 data.zip 을 '$DEST' 에 두고:  cd '$DEST' && unzip data.zip" >&2
  exit 1
fi

echo "[3/4] 압축 해제: $ZIP"
unzip -q -o "$ZIP"

echo "[4/4] 검증"
# data.zip 은 최상위에 data/ 를 만든다. (일부 배포는 test_instruction 을 바로 두기도 함)
if [ -d "data/test_instruction" ]; then
  TARGET="data/test_instruction"
elif [ -d "test_instruction" ]; then
  TARGET="test_instruction"
else
  echo "!!! 압축 해제 후 test_instruction 을 찾지 못했습니다." >&2
  echo "!!! zip 내부 구조 확인:  unzip -l '$ZIP' | grep -i instruction | head" >&2
  exit 1
fi

echo "완료. $DEST/$TARGET 파일:"
ls -1 "$TARGET" | grep -iE 'G[123]' || ls -1 "$TARGET" | head
echo ""
echo "다음: config.yaml 의 paths.toolbench_root='$DEST' 확인 후"
echo "  python src/m1_data_prep.py --config config.yaml"
