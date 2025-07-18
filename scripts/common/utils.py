"""共通ユーティリティ関数"""
import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv


def setup_environment():
    """環境変数と設定の読み込み"""
    env_path = Path(__file__).parent.parent.parent / "config" / ".env"
    load_dotenv(env_path)

    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Dict[str, Any], script_name: str = None):
    """ロギングの設定"""
    from datetime import datetime
    import inspect

    log_config = config.get('logging', {})

    if script_name is None:
        frame = inspect.stack()[1]
        script_path = Path(frame.filename)
        script_name = script_path.stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"

    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / log_filename

    # 重複防止のため既存のハンドラをクリア
    logger = logging.getLogger()
    logger.handlers.clear()

    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

    logger = logging.getLogger(__name__)
    logger.info(f"ログファイル: {log_file}")

    return logger


def get_vault_files(vault_path: Path, exclude_patterns: List[str] = None) -> List[Path]:
    """Vault内のMarkdownファイルを取得"""
    if not vault_path.exists():
        raise ValueError(f"Vault path does not exist: {vault_path}")

    exclude_patterns = exclude_patterns or []

    md_files = []
    for file_path in vault_path.rglob("*.md"):
        relative_path = file_path.relative_to(vault_path)
        should_exclude = False

        for pattern in exclude_patterns:
            if pattern.startswith("*") and pattern.endswith("*"):
                if pattern[1:-1] in str(relative_path):
                    should_exclude = True
                    break
            elif pattern.endswith("/*"):
                if str(relative_path).startswith(pattern[:-2]):
                    should_exclude = True
                    break
            elif pattern.startswith("."):
                if any(part.startswith(".") for part in relative_path.parts):
                    should_exclude = True
                    break
            elif pattern in str(relative_path):
                should_exclude = True
                break

        if not should_exclude:
            md_files.append(file_path)

    return sorted(md_files)


def format_file_info(file_path: Path, vault_path: Path) -> Dict[str, str]:
    """ファイル情報をフォーマット"""
    return {
        'absolute_path': str(file_path),
        'relative_path': str(file_path.relative_to(vault_path)),
        'name': file_path.stem,
        'directory': str(file_path.parent.relative_to(vault_path))
    }
