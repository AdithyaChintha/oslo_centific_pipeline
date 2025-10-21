#!/usr/bin/env python3
"""
SharePoint Utilities Module

This module provides shared utilities for SharePoint operations across all scripts.
It consolidates authentication, file management, and common SharePoint operations
to eliminate code duplication and ensure consistency.

Key Components:
- SharePointAuthenticator: Handles Microsoft Graph API authentication
- SharePointFileManager: Manages file operations (upload, download, list)
- SharePointConfig: Configuration dataclass for SharePoint settings

Author: Shared utilities for SharePoint automation
Date: October 2025
"""

import os
import sys
import json
import time
import yaml
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Import resumable upload manager
try:
    from .resumable_upload_manager import ResumableUploadManager
except ImportError:
    from resumable_upload_manager import ResumableUploadManager

# Get logger
logger = logging.getLogger(__name__)

@dataclass
class SharePointConfig:
    """
    Configuration class for SharePoint connection settings.
    
    This dataclass consolidates all SharePoint-related configuration
    parameters used across different scripts.
    """
    # Azure AD credentials
    tenant_id: str
    client_id: str  
    client_secret: str
    
    # SharePoint site details
    site_id: str
    drive_id: str
    
    # Folder paths (optional - scripts can provide their own)
    input_folder_path: Optional[str] = None
    output_folder_path: Optional[str] = None
    delivery_folder_path: Optional[str] = None
    jsonl_source_folder_path: Optional[str] = None
    
    # Local paths (optional - scripts can provide their own)
    excel_source_dir: Optional[str] = None
    download_dir: Optional[str] = None
    jsonl_output_dir: Optional[str] = None
    upload_state_file: Optional[str] = None
    
    # Processing settings
    skip_newest_files: int = 1
    max_files_per_run: int = 50
    
    # API settings
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 5

class SharePointAuthenticator:
    """
    Handles Microsoft Graph API authentication for SharePoint access.
    
    This class manages OAuth2 Client Credentials Flow authentication,
    token caching, and automatic token refresh for SharePoint operations.
    """
    
    def __init__(self, tenant_id: str, client_id: str, client_secret: str, timeout_seconds: int = 30):
        """
        Initialize authenticator with Azure AD credentials.
        
        Args:
            tenant_id: Azure AD tenant ID
            client_id: Azure AD app registration client ID  
            client_secret: Azure AD app registration client secret
            timeout_seconds: Request timeout in seconds
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_seconds = timeout_seconds
        self.access_token = None
        self.token_expires_at = None
        
        logger.debug(f"SharePoint authenticator initialized for tenant: {tenant_id}")
    
    def get_access_token(self) -> Optional[str]:
        """
        Get a valid access token for Microsoft Graph API.
        
        Uses OAuth2 Client Credentials Flow for server-to-server authentication.
        Automatically handles token caching and refresh.
        
        Returns:
            Valid access token string or None if authentication failed
        """
        # Check if we have a valid cached token (with 5-minute buffer)
        if (self.access_token and self.token_expires_at and 
            datetime.now() < self.token_expires_at - timedelta(minutes=5)):
            logger.debug("Using cached access token")
            return self.access_token
        
        logger.info("Requesting new access token from Microsoft Graph API")
        
        # OAuth2 Client Credentials Flow endpoint
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        
        # Prepare authentication request
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'https://graph.microsoft.com/.default',
            'grant_type': 'client_credentials'
        }
        
        try:
            # Request access token
            response = requests.post(
                token_url,
                headers=headers,
                data=data,
                timeout=self.timeout_seconds
            )
            response.raise_for_status()
            
            # Parse response
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
            
            # Calculate expiration time
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info("Successfully obtained access token")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get access token: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid token response format: {e}")
            return None

class SharePointFileManager:
    """
    Handles SharePoint file operations via Microsoft Graph API.
    
    This class provides methods for common SharePoint file operations:
    - Listing files in folders
    - Downloading files
    - Uploading files
    - Managing folder operations
    """
    
    def __init__(self, authenticator: SharePointAuthenticator, site_id: str, drive_id: str, 
                 timeout: int = 30, max_retries: int = 3, retry_delay: int = 5,
                 resumable_upload_state_dir: Optional[str] = None):
        """
        Initialize file manager with authenticator and site details.
        
        Args:
            authenticator: SharePoint authenticator instance
            site_id: SharePoint site ID
            drive_id: SharePoint drive ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Delay between retries in seconds
            resumable_upload_state_dir: Directory for resumable upload state (optional)
        """
        self.authenticator = authenticator
        self.site_id = site_id
        self.drive_id = drive_id
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create base API URLs
        self.base_api_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
        self.drive_api_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}"
        
        # Initialize resumable upload manager
        upload_state_dir = resumable_upload_state_dir or "/tmp/sharepoint_upload_sessions"
        self.resumable_upload_manager = ResumableUploadManager(state_dir=upload_state_dir)
        
        logger.debug(f"SharePoint file manager initialized for site: {site_id}")
    
    def _make_authenticated_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Make an authenticated request to Microsoft Graph API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            url: Full API URL
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Response object or None if all retries failed
        """
        token = self.authenticator.get_access_token()
        if not token:
            logger.error("Cannot make request: no valid access token")
            return None
        
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        kwargs['headers'] = headers
        
        # Add default timeout
        kwargs.setdefault('timeout', self.timeout)
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")
                response = requests.request(method, url, **kwargs)
                
                # Check for authentication errors
                if response.status_code == 401:
                    logger.warning("Access token expired, refreshing...")
                    self.authenticator.access_token = None  # Force token refresh
                    token = self.authenticator.get_access_token()
                    if token:
                        headers['Authorization'] = f'Bearer {token}'
                        kwargs['headers'] = headers
                        continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff: wait longer after each failure
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        logger.error(f"All {self.max_retries} attempts failed for {method} {url}")
        return None
    
    def get_folder_id(self, folder_path: str) -> Optional[str]:
        """
        Get SharePoint folder ID by path for use in file operations.
        
        Args:
            folder_path: SharePoint folder path (e.g., "/Team/Projects/Folder")
            
        Returns:
            Folder ID string or None if folder not found
        """
        logger.info(f"Looking up folder ID for: {folder_path}")
        
        # URL encode the folder path
        import urllib.parse
        encoded_path = urllib.parse.quote(folder_path)
        
        # Use drive API with path
        url = f"{self.drive_api_url}/root:{encoded_path}"
        
        response = self._make_authenticated_request('GET', url)
        if not response:
            logger.error(f"Failed to get folder info for: {folder_path}")
            return None
        
        try:
            folder_info = response.json()
            folder_id = folder_info.get('id')
            
            if folder_id:
                logger.info(f"Found folder ID: {folder_id}")
                return folder_id
            else:
                logger.error(f"No folder ID in response for: {folder_path}")
                return None
                
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid folder response format: {e}")
            return None
    
    def list_files_in_folder(self, folder_id: str, file_extensions: Optional[List[str]] = None) -> List[Dict]:
        """
        List all files in a SharePoint folder, optionally filtered by extensions.
        
        Args:
            folder_id: SharePoint folder ID
            file_extensions: List of file extensions to filter by (e.g., ['.xlsx', '.jsonl'])
            
        Returns:
            List of file information dictionaries
        """
        logger.info(f"Listing files in folder: {folder_id}")
        
        url = f"{self.drive_api_url}/items/{folder_id}/children"
        
        response = self._make_authenticated_request('GET', url)
        if not response:
            logger.error(f"Failed to list files in folder: {folder_id}")
            return []
        
        try:
            folder_contents = response.json()
            files = []
            
            for item in folder_contents.get('value', []):
                # Only include files, not folders
                if 'file' in item:
                    file_name = item.get('name', '')
                    
                    # Apply file extension filter if provided
                    if file_extensions:
                        if not any(file_name.lower().endswith(ext.lower()) for ext in file_extensions):
                            continue
                    
                    file_info = {
                        'id': item.get('id'),
                        'name': file_name,
                        'size': item.get('size', 0),
                        'modified': item.get('lastModifiedDateTime'),
                        'download_url': item.get('@microsoft.graph.downloadUrl'),
                        'created': item.get('createdDateTime'),
                        'modified_by': item.get('lastModifiedBy', {}).get('user', {}).get('displayName', 'Unknown')
                    }
                    files.append(file_info)
            
            logger.info(f"Found {len(files)} files in folder")
            return files
            
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid folder contents response: {e}")
            return []
    
    def download_file(self, download_url: str, local_path: str) -> bool:
        """
        Download a file from SharePoint using direct download URL.
        
        Args:
            download_url: Direct download URL from SharePoint
            local_path: Local path where file should be saved
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file content (no authentication needed for direct download URLs)
            response = requests.get(download_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Write file content
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(local_path)
            logger.debug(f"Downloaded file: {os.path.basename(local_path)} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file to {local_path}: {e}")
            return False
    
    def download_file_by_id(self, file_id: str, local_path: str) -> bool:
        """
        Download a file from SharePoint using file ID.
        
        Args:
            file_id: SharePoint file ID
            local_path: Local path where file should be saved
            
        Returns:
            True if download successful, False otherwise
        """
        url = f"{self.drive_api_url}/items/{file_id}/content"
        
        response = self._make_authenticated_request('GET', url)
        if not response:
            return False
        
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Write file content
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(local_path)
            logger.debug(f"Downloaded file by ID: {os.path.basename(local_path)} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file by ID to {local_path}: {e}")
            return False
    
    def upload_file(self, local_file_path: str, sharepoint_filename: str, folder_id: str, 
                   force_resumable: bool = False) -> bool:
        """
        Upload a file to SharePoint folder with automatic resumable upload for large files.
        
        Args:
            local_file_path: Path to local file to upload
            sharepoint_filename: Desired filename in SharePoint
            folder_id: SharePoint folder ID where file should be uploaded
            force_resumable: Force resumable upload even for small files (for testing)
            
        Returns:
            True if upload successful, False otherwise
        """
        if not os.path.exists(local_file_path):
            logger.error(f"Local file not found: {local_file_path}")
            return False
        
        file_size = os.path.getsize(local_file_path)
        logger.info(f"Uploading file: {sharepoint_filename} ({file_size:,} bytes)")
        
        # Use resumable upload for files larger than 4MB or if forced
        if file_size > 4 * 1024 * 1024 or force_resumable:  # 4MB
            logger.info("Using resumable upload (large file or forced)")
            return self._upload_large_file(local_file_path, sharepoint_filename, folder_id)
        else:
            logger.info("Using simple upload (small file)")
            return self._upload_small_file(local_file_path, sharepoint_filename, folder_id)
    
    def get_resumable_upload_progress(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> Optional[Dict]:
        """
        Get progress information for a resumable upload.
        
        Args:
            local_file_path: Local file path
            sharepoint_filename: SharePoint filename
            folder_id: SharePoint folder ID
            
        Returns:
            Progress dictionary or None if no active session
        """
        try:
            session_id = self.resumable_upload_manager._generate_session_id(
                local_file_path, sharepoint_filename, folder_id
            )
            session = self.resumable_upload_manager._load_session(session_id)
            
            if session and not session.completed:
                return self.resumable_upload_manager.get_upload_progress(session)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting upload progress: {e}")
            return None
    
    def cancel_resumable_upload(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Cancel an active resumable upload session.
        
        Args:
            local_file_path: Local file path
            sharepoint_filename: SharePoint filename  
            folder_id: SharePoint folder ID
            
        Returns:
            True if session was cancelled, False otherwise
        """
        try:
            session_id = self.resumable_upload_manager._generate_session_id(
                local_file_path, sharepoint_filename, folder_id
            )
            return self.resumable_upload_manager._delete_session(session_id)
            
        except Exception as e:
            logger.error(f"Error cancelling upload: {e}")
            return False
    
    def list_resumable_uploads(self) -> List[Dict]:
        """
        List all active resumable upload sessions.
        
        Returns:
            List of active session progress dictionaries
        """
        try:
            return self.resumable_upload_manager.list_active_sessions()
        except Exception as e:
            logger.error(f"Error listing resumable uploads: {e}")
            return []
    
    def cleanup_resumable_uploads(self) -> int:
        """
        Clean up completed and expired resumable upload sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            return self.resumable_upload_manager.cleanup_completed_sessions()
        except Exception as e:
            logger.error(f"Error cleaning up uploads: {e}")
            return 0
    
    def _upload_small_file(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Upload a small file (< 4MB) to SharePoint using simple upload.
        
        Args:
            local_file_path: Local file path
            sharepoint_filename: SharePoint filename
            folder_id: Target folder ID
            
        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Prepare upload URL
            import urllib.parse
            encoded_filename = urllib.parse.quote(sharepoint_filename)
            url = f"{self.drive_api_url}/items/{folder_id}:/{encoded_filename}:/content"
            
            # Read file content
            with open(local_file_path, 'rb') as f:
                file_content = f.read()
            
            # Upload file
            response = self._make_authenticated_request(
                'PUT', 
                url, 
                data=file_content,
                headers={'Content-Type': 'application/octet-stream'}
            )
            
            if response and response.status_code in [200, 201]:
                logger.info(f"Successfully uploaded: {sharepoint_filename}")
                return True
            else:
                logger.error(f"Upload failed for: {sharepoint_filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error uploading small file {sharepoint_filename}: {e}")
            return False
    
    def _upload_large_file(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Upload a large file (>= 4MB) to SharePoint using TRUE resumable upload.
        
        This method provides genuine resumable upload functionality:
        - Session persistence across script runs
        - Progress tracking at chunk level
        - Automatic resume from interruption point
        - Session validation and recovery
        
        Args:
            local_file_path: Local file path
            sharepoint_filename: SharePoint filename  
            folder_id: Target folder ID
            
        Returns:
            True if upload successful, False otherwise
        """
        logger.info(f"Starting resumable upload: {sharepoint_filename}")
        
        try:
            # Step 1: Create or resume upload session
            session = self.resumable_upload_manager.create_or_resume_session(
                file_path=local_file_path,
                sharepoint_filename=sharepoint_filename,
                folder_id=folder_id,
                file_manager=self
            )
            
            if not session:
                logger.error("Failed to create or resume upload session")
                return False
            
            # Step 2: Get current progress
            progress = self.resumable_upload_manager.get_upload_progress(session)
            logger.info(f"Upload progress: {progress['uploaded_chunks']}/{progress['total_chunks']} chunks ({progress['progress_percent']:.1f}%)")
            
            # Step 3: Upload remaining chunks
            while not session.completed:
                chunk_success, upload_complete = self.resumable_upload_manager.upload_next_chunk(session)
                
                if not chunk_success:
                    logger.error("Failed to upload chunk")
                    return False
                
                if upload_complete:
                    logger.info(f"✅ Successfully completed resumable upload: {sharepoint_filename}")
                    
                    # Clean up completed session
                    self.resumable_upload_manager._delete_session(session.session_id)
                    return True
                
                # Log progress periodically
                progress = self.resumable_upload_manager.get_upload_progress(session)
                if progress['uploaded_chunks'] % 10 == 0:  # Every 10 chunks
                    logger.info(f"Upload progress: {progress['uploaded_chunks']}/{progress['total_chunks']} chunks ({progress['progress_percent']:.1f}%)")
            
            # Should not reach here, but handle completed session
            logger.info(f"✅ Upload session already completed: {sharepoint_filename}")
            self.resumable_upload_manager._delete_session(session.session_id)
            return True
            
        except Exception as e:
            logger.error(f"Error in resumable upload for {sharepoint_filename}: {e}")
            # Don't delete session on error - allow for retry/resume
            return False

def load_sharepoint_config(config_path: str, **overrides) -> SharePointConfig:
    """
    Load SharePoint configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to the YAML configuration file
        **overrides: Optional parameter overrides
        
    Returns:
        SharePointConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required configuration keys are missing
    """
    logger.info(f"Loading SharePoint configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    try:
        # Extract configuration sections
        azure_ad = config_data['azure_ad']
        sharepoint = config_data['sharepoint']
        local_paths = config_data.get('local_paths', {})
        processing = config_data.get('processing', {})
        api_config = config_data.get('api', {})
        
        # Create config with defaults and overrides
        config_params = {
            # Required parameters
            'tenant_id': azure_ad['tenant_id'],
            'client_id': azure_ad['client_id'],
            'client_secret': azure_ad['client_secret'],
            'site_id': sharepoint['site_id'],
            'drive_id': sharepoint['drive_id'],
            
            # Optional SharePoint paths
            'input_folder_path': sharepoint.get('input_folder_path'),
            'output_folder_path': sharepoint.get('output_folder_path'),
            'delivery_folder_path': sharepoint.get('delivery_folder_path'),
            'jsonl_source_folder_path': sharepoint.get('jsonl_source_folder_path'),
            
            # Optional local paths
            'excel_source_dir': local_paths.get('excel_source_dir'),
            'download_dir': local_paths.get('download_dir'),
            'jsonl_output_dir': local_paths.get('jsonl_output_dir'),
            'upload_state_file': local_paths.get('upload_state_file'),
            
            # Processing settings
            'skip_newest_files': processing.get('skip_newest_files', 1),
            'max_files_per_run': processing.get('max_files_per_run', 50),
            
            # API settings
            'timeout_seconds': api_config.get('timeout_seconds', 300),
            'retry_attempts': api_config.get('retry_attempts', 3),
            'retry_delay_seconds': api_config.get('retry_delay_seconds', 5)
        }
        
        # Apply any overrides
        config_params.update(overrides)
        
        return SharePointConfig(**config_params)
        
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")

def create_sharepoint_manager(config: SharePointConfig, 
                             resumable_upload_state_dir: Optional[str] = None) -> Tuple[SharePointAuthenticator, SharePointFileManager]:
    """
    Create SharePoint authenticator and file manager from configuration.
    
    Args:
        config: SharePoint configuration object
        resumable_upload_state_dir: Directory for resumable upload state (optional)
        
    Returns:
        Tuple of (authenticator, file_manager)
    """
    # Create authenticator
    authenticator = SharePointAuthenticator(
        tenant_id=config.tenant_id,
        client_id=config.client_id,
        client_secret=config.client_secret,
        timeout_seconds=config.timeout_seconds
    )
    
    # Create file manager with resumable upload support
    file_manager = SharePointFileManager(
        authenticator=authenticator,
        site_id=config.site_id,
        drive_id=config.drive_id,
        timeout=config.timeout_seconds,
        max_retries=config.retry_attempts,
        retry_delay=config.retry_delay_seconds,
        resumable_upload_state_dir=resumable_upload_state_dir
    )
    
    return authenticator, file_manager
