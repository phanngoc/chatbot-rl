#!/usr/bin/env python3
"""
Script để chạy Memory Visualizer App
Automatically setup và launch Streamlit visualization dashboard
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'scikit-learn', 'networkx', 'umap-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(missing_packages):
    """Install missing dependencies"""
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables"""
    # Set default ChromaDB path nếu chưa có
    if not os.getenv('CHROMA_DB_PATH'):
        default_path = Path(__file__).parent / 'src' / 'data' / 'chroma_db'
        os.environ['CHROMA_DB_PATH'] = str(default_path)
        print(f"🗄️  ChromaDB path set to: {default_path}")
    
    # OpenAI API key reminder
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OpenAI API key not found. Some features may be limited.")
        print("   Set OPENAI_API_KEY environment variable for full functionality.")

def run_streamlit_app(port=8501, auto_open=True):
    """Launch Streamlit app"""
    visualizer_path = Path(__file__).parent / 'memory_visualizer.py'
    
    if not visualizer_path.exists():
        print(f"❌ Visualizer app not found at: {visualizer_path}")
        return False
    
    # Streamlit command
    cmd = [
        'streamlit', 'run', str(visualizer_path),
        '--server.port', str(port),
        '--server.headless', str(not auto_open).lower(),
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"🚀 Starting Memory Visualizer at: http://localhost:{port}")
    print("   Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent)
        return True
    except KeyboardInterrupt:
        print("\n👋 Memory Visualizer stopped.")
        return True
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Memory Visualizer Launcher')
    parser.add_argument('--port', type=int, default=8501, help='Port to run Streamlit on')
    parser.add_argument('--no-auto-open', action='store_true', help='Don\'t auto-open browser')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency check')
    
    args = parser.parse_args()
    
    print("🧠 Memory Visualizer Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not args.skip_deps:
        print("🔍 Checking dependencies...")
        missing = check_dependencies()
        
        if missing:
            print(f"❌ Missing packages: {', '.join(missing)}")
            install_choice = input("📦 Install missing packages? (y/n): ").lower().strip()
            
            if install_choice == 'y':
                if not install_dependencies(missing):
                    print("❌ Failed to install dependencies. Exiting.")
                    return 1
            else:
                print("⚠️  Continuing without installing. Some features may not work.")
        else:
            print("✅ All dependencies are installed!")
    
    # Setup environment
    print("⚙️  Setting up environment...")
    setup_environment()
    
    # Run app
    success = run_streamlit_app(
        port=args.port, 
        auto_open=not args.no_auto_open
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
