"""e5 임베딩 래퍼 (intfloat/multilingual-e5-large).

계약:
  embed_queries(texts) -> np.ndarray   # prefix "query: " 적용
  embed_passages(texts) -> np.ndarray  # prefix "passage: " 적용
  둘 다 mean pooling + L2 normalize.

주의: e5 prefix 규칙 필수. 빠뜨리면 성능 급락.
구현: Claude Code (sentence-transformers 또는 transformers 직접)
"""
# TODO(Claude Code)
