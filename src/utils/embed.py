"""e5 임베딩 래퍼 (intfloat/multilingual-e5-large).

계약:
  embed_queries(texts, config) -> np.ndarray   # prefix "query: " 적용
  embed_passages(texts, config) -> np.ndarray  # prefix "passage: " 적용
  둘 다 mean pooling + L2 normalize.

주의: e5 prefix 규칙 필수. 빠뜨리면 성능 급락.
구현: Claude Code (transformers 직접 사용, v5 호환)

계약 관련 메모:
  - 스텁 docstring 의 원 시그니처는 `embed_queries(texts)` 였으나, 모델 id/디바이스/
    배치크기를 하드코딩하지 않으려면 config 가 필요하다(CLAUDE.md 규칙 4).
    따라서 config 를 인자로 받는다. prefix·pooling·normalize 계약은 그대로 지킨다.
  - 모델 로딩은 비싸므로 model id 기준으로 프로세스 내 캐싱한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# e5 prefix 규칙 (변경 금지). query 는 "query: ", 문서는 "passage: ".
PREFIX_QUERY = "query: "
PREFIX_PASSAGE = "passage: "

# DECISION NEEDED: e5 토큰 max_length=512.
#   근거: multilingual-e5-large 는 XLM-R 기반으로 최대 512 토큰. 실험 파라미터가
#   아니라 모델 구조 상한이라 config 화하지 않는다. 문서/example 이 길면 절단된다.
_MAX_LENGTH = 512

# model id 기준 임베더 캐시.
_CACHE: dict[str, "E5Embedder"] = {}


def _average_pool(last_hidden_state, attention_mask):
    """attention mask 기반 mean pooling (e5 권장 방식)."""
    import torch  # 지연 임포트: config-only 경로에서 torch 미설치여도 로드되게.

    mask = attention_mask[..., None].bool()
    masked = last_hidden_state.masked_fill(~mask, 0.0)
    summed = masked.sum(dim=1)
    counts = attention_mask.sum(dim=1)[..., None].clamp(min=1)
    return summed / counts


class E5Embedder:
    """intfloat/multilingual-e5-large 임베더. mean pooling + L2 normalize.

    임베딩은 float32 로 계산한다 (다운스트림 dtype 과 무관).
    DECISION NEEDED: 임베딩 precision=float32.
      근거: cosine 검색은 pooling/정규화 수치 안정성이 중요하고, e5-large 는
      작아 fp32 로도 부담이 없다. 다운스트림 생성 dtype(bfloat16)과 분리한다.
    """

    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_id = config["models"]["embedder"]
        hardware = config.get("hardware", {})
        requested = str(hardware.get("device", "cpu"))
        # cuda 요청이지만 사용 불가면 cpu 로 안전 강등 (서버 외 환경에서도 로딩되게).
        if requested.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = requested
        self.batch_size = int(hardware.get("batch_size_embed", 32))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # 임베딩은 encoder(AutoModel). dtype 인자 없이 기본 fp32 로 로드 후 device 이동
        # (transformers v4/v5 dtype 인자명 차이를 피한다; CODING_NOTES 참고).
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.eval()
        self.model.to(self.device)

    def _embed(self, texts: list[str], prefix: str) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if len(texts) == 0:
            # e5-large hidden size = 1024. 빈 입력에 대해 형태를 지키는 빈 배열 반환.
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        out_batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [prefix + t for t in texts[start : start + self.batch_size]]
                enc = self.tokenizer(
                    batch,
                    max_length=_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**enc)
                pooled = _average_pool(outputs.last_hidden_state, enc["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                out_batches.append(pooled.float().cpu().numpy())
        return np.concatenate(out_batches, axis=0)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts), PREFIX_QUERY)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._embed(list(texts), PREFIX_PASSAGE)


def get_embedder(config: dict[str, Any]) -> E5Embedder:
    """config.models.embedder 기준으로 임베더를 캐시·반환한다."""
    model_id = config["models"]["embedder"]
    if model_id not in _CACHE:
        _CACHE[model_id] = E5Embedder(config)
    return _CACHE[model_id]


def embed_queries(texts: list[str], config: dict[str, Any]) -> np.ndarray:
    """쿼리 임베딩. prefix "query: " + mean pooling + L2 normalize."""
    return get_embedder(config).embed_queries(texts)


def embed_passages(texts: list[str], config: dict[str, Any]) -> np.ndarray:
    """문서 임베딩. prefix "passage: " + mean pooling + L2 normalize."""
    return get_embedder(config).embed_passages(texts)
