#!/usr/bin/env python3
"""
ChromaDBをクリーンアップするスクリプト
"""
import sys
from pathlib import Path
import shutil
import click

sys.path.append(str(Path(__file__).parent))
from common.utils import setup_environment, setup_logging
from common.db import VectorDB


@click.command()
@click.option('--all', is_flag=True, help='すべてのデータを削除')
@click.option('--remove', help='特定のコレクションを削除（"all"で全コレクション削除）')
@click.option('--list', 'list_only', is_flag=True, help='コレクション一覧を表示')
def clean(all, remove, list_only):
    """ChromaDBのクリーンアップ"""
    config = setup_environment()
    logger = setup_logging(config)
    
    db_path = Path(config['database']['persist_directory'])
    
    if list_only:
        db = VectorDB(str(db_path))
        collections = db.list_collections()
        
        if collections:
            logger.info("現在のコレクション:")
            for col in collections:
                stats = db.get_collection_stats(col)
                logger.info(f"  - {col}: {stats.get('document_count', 0)}件")
        else:
            logger.info("コレクションが見つかりません")
        return
    
    if all:
        if click.confirm('本当にすべてのデータを削除しますか？'):
            logger.info("すべてのデータを削除します...")
            
            if db_path.exists():
                shutil.rmtree(db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("✅ データベースをクリーンアップしました")
        else:
            logger.info("キャンセルしました")
            
    elif remove:
        db = VectorDB(str(db_path))
        
        if remove.lower() == 'all':
            collections = db.list_collections()
            if not collections:
                logger.info("削除するコレクションがありません")
                return
                
            logger.info(f"以下の{len(collections)}個のコレクションを削除します:")
            for col in collections:
                logger.info(f"  - {col}")
            
            if click.confirm('本当にすべてのコレクションを削除しますか？'):
                for col in collections:
                    try:
                        db.delete_collection(col)
                        logger.info(f"✅ コレクション '{col}' を削除しました")
                    except Exception as e:
                        logger.error(f"コレクション '{col}' の削除エラー: {e}")
            else:
                logger.info("キャンセルしました")
        else:
            try:
                db.delete_collection(remove)
                logger.info(f"✅ コレクション '{remove}' を削除しました")
            except Exception as e:
                logger.error(f"削除エラー: {e}")
    
    else:
        logger.info("使用方法:")
        logger.info("  python scripts/clean_db.py --list")
        logger.info("  python scripts/clean_db.py --all")
        logger.info("  python scripts/clean_db.py --remove all")
        logger.info("  python scripts/clean_db.py --remove obsidian_gemini")


if __name__ == "__main__":
    clean()