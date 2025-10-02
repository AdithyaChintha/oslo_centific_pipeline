import os
import logging
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class MovementMetadataLoader:
    """
    Loads and caches movement metadata from S3 parquet file.
    """

    def __init__(self, s3_client, bucket: str, s3_key: str, cache_dir: str = "./cache"):
        """
        Initialize movement metadata loader.

        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            s3_key: S3 key to parquet file
            cache_dir: Local directory to cache parquet file
        """
        self.s3_client = s3_client
        self.bucket = bucket
        self.s3_key = s3_key
        self.cache_dir = cache_dir
        self.df = None
        self._lookup_dict = None

        os.makedirs(cache_dir, exist_ok=True)

    def load(self, force_reload: bool = False) -> bool:
        """
        Download and load parquet file from S3.

        Args:
            force_reload: If True, download even if already cached

        Returns:
            bool: True if successful, False otherwise
        """
        local_path = os.path.join(self.cache_dir, "movement_metadata.parquet")

        # Download if not cached or force reload
        if force_reload or not os.path.exists(local_path):
            try:
                logger.info(f"Downloading movement metadata from s3://{self.bucket}/{self.s3_key}")
                self.s3_client.download_file(self.bucket, self.s3_key, local_path)
                logger.info(f"Downloaded movement metadata to {local_path}")
            except Exception as e:
                logger.error(f"Failed to download movement metadata: {e}")
                return False
        else:
            logger.info(f"Using cached movement metadata from {local_path}")

        # Load parquet into DataFrame
        try:
            self.df = pd.read_parquet(local_path)
            logger.info(f"Loaded {len(self.df)} movement metadata records")

            # Build lookup dictionary for fast access: s3_key -> row data
            self._build_lookup_dict()

            return True
        except Exception as e:
            logger.error(f"Failed to load parquet file: {e}")
            return False

    def _build_lookup_dict(self):
        """Build dictionary for O(1) lookup by s3_key"""
        if self.df is None:
            self._lookup_dict = {}
            return

        def sanitize_value(val):
            """Convert NaN to None for JSON compatibility"""
            if pd.isna(val):
                return None
            return val

        self._lookup_dict = {}
        for _, row in self.df.iterrows():
            s3_key = row.get('s3_key')
            if s3_key:
                # Store movement metadata fields only, converting NaN to None
                self._lookup_dict[s3_key] = {
                    'has_movement': sanitize_value(row.get('has_movement')),
                    'movement_type_1': sanitize_value(row.get('movement_type_1')),
                    'confidence_1': sanitize_value(row.get('confidence_1')),
                    'movement_type_2': sanitize_value(row.get('movement_type_2')),
                    'confidence_2': sanitize_value(row.get('confidence_2')),
                    'movement_type_3': sanitize_value(row.get('movement_type_3')),
                    'confidence_3': sanitize_value(row.get('confidence_3')),
                    'movement_type_4': sanitize_value(row.get('movement_type_4')),
                    'confidence_4': sanitize_value(row.get('confidence_4')),
                    'movement_type_5': sanitize_value(row.get('movement_type_5')),
                    'confidence_5': sanitize_value(row.get('confidence_5'))
                }

        logger.info(f" Built movement metadata lookup with {len(self._lookup_dict)} entries")

    def get_movement_metadata(self, s3_key: str) -> Optional[Dict]:
        """
        Get movement metadata for a given S3 key.

        Args:
            s3_key: Full S3 URI (e.g., "s3://bucket/path/to/clip.mov")

        Returns:
            dict with movement metadata fields, or None if not found
        """
        if self._lookup_dict is None:
            logger.warning("Movement metadata not loaded")
            return None

        metadata = self._lookup_dict.get(s3_key)

        if metadata:
            logger.debug(f"Found movement metadata for {s3_key}")
        else:
            logger.debug(f" No movement metadata found for {s3_key}")

        return metadata

    def is_loaded(self) -> bool:
        """Check if metadata is loaded and ready"""
        return self.df is not None and self._lookup_dict is not None
