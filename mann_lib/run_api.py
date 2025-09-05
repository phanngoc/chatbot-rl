#!/usr/bin/env python3
"""
Script để chạy MANN API Server
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from standalone_mann.mann_api import run_server
from standalone_mann.mann_config import MANNConfig

if __name__ == "__main__":
    config = MANNConfig()
    config.update_from_env()
    
    print(f"🚀 Starting MANN API server on {config.api_host}:{config.api_port}")
    print(f"📊 Monitoring: {'Enabled' if config.enable_monitoring else 'Disabled'}")
    print(f"📱 Pager: {'Enabled' if config.enable_pager else 'Disabled'}")
    print(f"💾 Data directory: {config.data_dir}")
    print(f"📝 Log level: {config.log_level}")
    
    run_server(config)
