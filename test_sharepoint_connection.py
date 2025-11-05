#!/usr/bin/env python3
"""
SharePoint Connection Test Script

Tests authentication and basic folder access to verify configuration.

Usage:
    python test_sharepoint_connection.py [--config CONFIG_FILE]
"""

import os
import sys
import yaml
import requests
from datetime import datetime

def load_config(config_path="config/tar_transfer_config.yaml"):
    """Load and validate configuration."""
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return None

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        azure_ad = config['azure_ad']
        sharepoint = config['sharepoint']

        # Check for placeholder values
        placeholders = []
        if "YOUR_" in azure_ad.get('tenant_id', ''):
            placeholders.append('tenant_id')
        if "YOUR_" in azure_ad.get('client_id', ''):
            placeholders.append('client_id')
        if "YOUR_" in azure_ad.get('client_secret', ''):
            placeholders.append('client_secret')
        if "YOUR_" in sharepoint.get('site_id', ''):
            placeholders.append('site_id')

        if placeholders:
            print(f"❌ Please fill out these configuration values: {', '.join(placeholders)}")
            return None

        return config

    except KeyError as e:
        print(f"❌ Missing configuration section: {e}")
        return None

def get_access_token(config):
    """Test authentication and get access token."""
    print("🔑 Testing authentication...")
    
    azure_ad = config['azure_ad']
    token_url = f"https://login.microsoftonline.com/{azure_ad['tenant_id']}/oauth2/v2.0/token"
    
    data = {
        'client_id': azure_ad['client_id'],
        'client_secret': azure_ad['client_secret'],
        'scope': 'https://graph.microsoft.com/.default',
        'grant_type': 'client_credentials'
    }
    
    try:
        response = requests.post(token_url, data=data, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        access_token = token_data.get('access_token')
        
        if access_token:
            print("✅ Authentication successful!")
            return access_token
        else:
            print("❌ No access token in response")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Authentication failed: {e}")
        return None

def test_site_access(config, token):
    """Test SharePoint site access."""
    print("🏢 Testing SharePoint site access...")
    
    site_id = config['sharepoint']['site_id']
    url = f"https://graph.microsoft.com/v1.0/sites/{site_id}"
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        site_data = response.json()
        site_name = site_data.get('displayName', 'Unknown')
        print(f"✅ Site access successful: {site_name}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Site access failed: {e}")
        return False

def test_folder_access(config, token, folder_path, folder_name):
    """Test access to a specific folder using drive ID."""
    print(f"📁 Testing {folder_name} folder access: {folder_path}")
    
    site_id = config['sharepoint']['site_id']
    drive_id = config['sharepoint']['drive_id']
    clean_path = folder_path.strip('/')
    
    if not clean_path:
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root"
    else:
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root:/{clean_path}"
    
    headers = {'Authorization': f'Bearer {token}'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        folder_data = response.json()
        folder_id = folder_data.get('id', 'Unknown')
        folder_name_actual = folder_data.get('name', 'Unknown')
        print(f"✅ {folder_name} folder access successful: {folder_name_actual}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ {folder_name} folder access failed: {e}")
        print(f"   Make sure folder exists: {folder_path}")
        return False

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SharePoint connection")
    parser.add_argument('--config', default='config/tar_transfer_config.yaml', help='Configuration file path')
    args = parser.parse_args()

    print("=" * 50)
    print("SharePoint Connection Test (TAR Transfer)")
    print("=" * 50)

    # Load configuration
    config = load_config(args.config)
    if not config:
        sys.exit(1)

    # Test authentication
    token = get_access_token(config)
    if not token:
        sys.exit(1)

    # Test site access
    if not test_site_access(config, token):
        sys.exit(1)

    # Test folder access
    sharepoint_config = config['sharepoint']
    folders_to_test = [
        (sharepoint_config.get('tar_source_folder_path', '/Uploads'), 'TAR Source')
    ]

    all_folders_ok = True
    for folder_path, folder_name in folders_to_test:
        if not test_folder_access(config, token, folder_path, folder_name):
            all_folders_ok = False

    print("=" * 50)
    if all_folders_ok:
        print("🎉 All tests passed! SharePoint configuration is working correctly.")
        print("\nNext steps:")
        print("1. Run TAR transfer script to download TAR files from SharePoint")
        print("2. Upload TAR files to Azure Blob Storage")
    else:
        print("⚠️  Some tests failed. Please check folder paths and permissions.")
    print("=" * 50)

if __name__ == '__main__':
    main()
