"""Config 로더. config.yaml을 읽고 ${var} 치환, 경로 확장.

계약:
  load_config(path) -> dict  (경로는 절대경로로 확장, ${output_dir} 등 치환)
구현: Claude Code

동작:
  1. YAML 로드.
  2. paths.* 안의 ${var} 참조를 반복 치환 (예: data_dir="${output_dir}/data").
  3. paths.* 를 expanduser + 절대경로로 확장.
  4. output_dir 하위 산출물 디렉터리(data/results/models) 자동 생성.

하드코딩 금지 원칙: 이 로더는 config.yaml의 값을 해석만 하고, 실험 파라미터를
자체적으로 정하지 않는다. 반환 dict가 모든 모듈의 단일 파라미터 출처다.
"""
from __future__ import annotations

import os
import re
from typing import Any

import yaml

# ${var} 형태 참조. var 이름은 영숫자/밑줄.
_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

# output_dir 하위로 자동 생성할 산출물 디렉터리 키 (config.paths 안의 이름).
_OUTPUT_SUBDIR_KEYS = ("data_dir", "results_dir", "models_dir")


def _interpolate_paths(paths: dict[str, Any]) -> dict[str, Any]:
    """paths dict 내부의 ${key} 참조를 같은 dict의 값으로 반복 치환.

    ${output_dir} 처럼 다른 paths 항목을 참조하는 경우만 지원한다 (config 구조 기준).
    치환이 더 이상 변하지 않을 때까지 (최대 depth) 반복해 중첩 참조도 해소한다.
    """
    resolved = dict(paths)
    # 참조 깊이 상한: paths 항목 수만큼이면 어떤 선형 참조 사슬도 해소된다.
    for _ in range(len(resolved) + 1):
        changed = False
        for key, val in resolved.items():
            if not isinstance(val, str):
                continue

            def _sub(m: re.Match) -> str:
                name = m.group(1)
                if name not in resolved:
                    # 알 수 없는 참조는 그대로 두고(치환 실패를 숨기지 않음) 원문 유지.
                    return m.group(0)
                return str(resolved[name])

            new_val = _VAR_RE.sub(_sub, val)
            if new_val != val:
                resolved[key] = new_val
                changed = True
        if not changed:
            break
    # 미해소 참조가 남아 있으면 조용히 통과시키지 않고 실패로 알린다.
    for key, val in resolved.items():
        if isinstance(val, str) and _VAR_RE.search(val):
            raise ValueError(
                f"config paths.{key}='{val}' 에 해소되지 않은 ${{...}} 참조가 있습니다."
            )
    return resolved


def load_config(path: str, *, make_dirs: bool = True) -> dict[str, Any]:
    """config.yaml 을 로드해 파싱·치환·경로확장한 dict 를 반환한다.

    Args:
      path: config.yaml 경로.
      make_dirs: True면 output_dir 하위 산출물 디렉터리를 생성한다.
                 (--smoke 처럼 쓰기 권한이 없을 때만 False 로.)

    Returns:
      해석된 config dict. paths.* 는 절대경로.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config.yaml 최상위가 매핑이 아닙니다: {type(cfg)}")

    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("config.yaml 에 'paths' 매핑이 없습니다.")

    # 1) ${...} 치환 → 2) expanduser → 3) 절대경로화.
    # DECISION NEEDED: output_dir 을 프로세스 cwd 기준 절대경로로 고정한다.
    #   근거: 파생 경로(data/results/models)가 cwd 변경과 무관하게 안정적이려면
    #   기준을 한 번 고정해야 한다. run_all.sh 는 레포 루트에서 실행되는 전제.
    paths = _interpolate_paths(paths)
    for key, val in list(paths.items()):
        if isinstance(val, str):
            paths[key] = os.path.abspath(os.path.expanduser(val))
    cfg["paths"] = paths

    if make_dirs:
        os.makedirs(paths["output_dir"], exist_ok=True)
        for key in _OUTPUT_SUBDIR_KEYS:
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)

    return cfg
