#!/usr/bin/env bash
# ============================================================
# 전체 실험 파이프라인. 각 단계 후 검증(gate)이 돌고, 실패 시 즉시 중단(exit 1)한다.
# 사용법: bash run_all.sh [--config config.yaml]
# 재실행: 중단된 지점을 고친 뒤 다시 실행. 각 스크립트는 산출물이 이미 있으면
#         건너뛰거나 덮어쓰기 정책을 따른다(각 스크립트 --force 참고).
# ============================================================
set -euo pipefail

CONFIG="${2:-config.yaml}"
PY="python"

log()  { echo -e "\n=== [$(date +%H:%M:%S)] $1 ==="; }
gate() {
  # gate <검증스크립트> <설명>
  log "GATE: $2"
  if ! $PY "$1" --config "$CONFIG"; then
    echo "!!! GATE FAILED: $2"
    echo "!!! 원인을 확인하고 수정한 뒤 재실행하세요. 파이프라인을 중단합니다."
    exit 1
  fi
  echo ">>> GATE PASSED: $2"
}

# ---------- M1: 데이터 준비 ----------
log "M1 데이터 준비"
$PY src/m1_data_prep.py --config "$CONFIG"
gate src/verify/verify_m1.py "M1 데이터 무결성 (gold 누락 0, description 결측 0, split 크기)"

# ---------- M2: example 검증 (생성 아님) ----------
# data/tools_examples.jsonl은 Claude Code가 작성 단계에서 커밋한 것을 사용.
# m2는 로드+누출 임베딩 계산만, 판정은 verify_m2.
log "M2 example 검증"
$PY src/m2_examples.py --config "$CONFIG"
gate src/verify/verify_m2.py "M2 example 누출 컷 (max_leak_sim <= threshold) + 커버리지"

# ---------- M3: retrieval + fusion 계수 ----------
log "M3 retrieval"
$PY src/m3_retrieval.py --config "$CONFIG"
gate src/verify/verify_m3.py "M3 Recall_all 검산 + fusion 계수 test 미사용"

# ---------- M4: classifier ----------
log "M4 classifier"
$PY src/m4_classifier.py --config "$CONFIG"
gate src/verify/verify_m4.py "M4 test 누출 0 + classifier_eval 생성"

# ---------- M5: 파일럿 (2B 파싱 검증) ----------
log "M5 파일럿"
$PY src/m5_pilot.py --config "$CONFIG"
gate src/verify/verify_m5.py "M5 파싱 성공률 검증 (미달 모델 자동 제외 기록)"

# ---------- M6: 전체 downstream + 분석 ----------
log "M6 전체 실행 + 분석"
$PY src/m6_downstream.py --config "$CONFIG"   # 파일럿 통과 조합 전체 실행
$PY src/m6_analysis.py --config "$CONFIG"      # 표 1·2·3, 그림 1, summary.csv
gate src/verify/verify_m6.py "M6 조합 누락 0 + 결과표 생성"

log "완료. 결과: outputs/results/summary.csv, 표·그림은 outputs/results/"
