#!/usr/bin/env python3
"""
Script để chạy MANN CLI Chatbot
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mann_chatbot import main

if __name__ == "__main__":
    asyncio.run(main())
