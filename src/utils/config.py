"""Config 로더. config.yaml을 읽고 ${var} 치환, 경로 확장.

계약:
  load_config(path) -> dict  (경로는 절대경로로 확장, ${output_dir} 등 치환)
구현: Claude Code
"""
# TODO(Claude Code): yaml 로드 + ${...} 치환 + os.path.expanduser + output_dir 하위 생성
