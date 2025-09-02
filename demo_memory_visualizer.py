#!/usr/bin/env python3
"""
Demo script cho Memory Visualizer
Tạo sample data và launch visualization app
"""

import json
import numpy as np
from datetime import datetime, timedelta
import subprocess
import sys
from pathlib import Path

def create_demo_data():
    """Tạo demo data với realistic memory examples"""
    
    demo_memories = [
        # Cooking memories
        {
            "id": "mem_001",
            "content": "User hỏi về cách nấu phở bò ngon",
            "context": "Cuộc trò chuyện về ẩm thực Việt Nam",
            "importance_score": 0.85,
            "access_count": 12,
            "tags": ["cooking", "vietnamese", "pho", "food"],
            "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
            "embedding": np.random.normal(0.2, 0.1, 384).tolist()
        },
        {
            "id": "mem_002", 
            "content": "Hướng dẫn làm bánh mì Việt Nam",
            "context": "Thảo luận về làm bánh và nướng",
            "importance_score": 0.75,
            "access_count": 8,
            "tags": ["cooking", "vietnamese", "bread", "baking"],
            "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
            "embedding": np.random.normal(0.15, 0.1, 384).tolist()
        },
        
        # Tech memories
        {
            "id": "mem_003",
            "content": "Giải thích về machine learning và neural networks",
            "context": "Hỏi đáp về AI và công nghệ",
            "importance_score": 0.92,
            "access_count": 20,
            "tags": ["tech", "ai", "ml", "education"],
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "embedding": np.random.normal(-0.3, 0.1, 384).tolist()
        },
        {
            "id": "mem_004",
            "content": "Hướng dẫn sử dụng Python cho data science",
            "context": "Tutorial về programming",
            "importance_score": 0.88,
            "access_count": 15,
            "tags": ["tech", "python", "data", "programming"],
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "embedding": np.random.normal(-0.25, 0.1, 384).tolist()
        },
        
        # Travel memories
        {
            "id": "mem_005",
            "content": "Kinh nghiệm du lịch Hà Nội và Sài Gòn",
            "context": "Tư vấn địa điểm du lịch",
            "importance_score": 0.70,
            "access_count": 6,
            "tags": ["travel", "vietnam", "hanoi", "saigon"],
            "timestamp": (datetime.now() - timedelta(days=7)).isoformat(),
            "embedding": np.random.normal(0.4, 0.1, 384).tolist()
        },
        {
            "id": "mem_006",
            "content": "Đề xuất lịch trình du lịch Đà Nẵng 3 ngày",
            "context": "Planning cho kỳ nghỉ",
            "importance_score": 0.65,
            "access_count": 4,
            "tags": ["travel", "vietnam", "danang", "planning"],
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "embedding": np.random.normal(0.35, 0.1, 384).tolist()
        },
        
        # Health memories
        {
            "id": "mem_007",
            "content": "Lời khuyên về chế độ ăn uống lành mạnh",
            "context": "Tư vấn sức khỏe và dinh dưỡng",
            "importance_score": 0.80,
            "access_count": 10,
            "tags": ["health", "nutrition", "diet", "wellness"],
            "timestamp": (datetime.now() - timedelta(days=4)).isoformat(),
            "embedding": np.random.normal(0.0, 0.1, 384).tolist()
        },
        {
            "id": "mem_008",
            "content": "Bài tập thể dục tại nhà cho người bận rộn",
            "context": "Fitness và sức khỏe",
            "importance_score": 0.72,
            "access_count": 7,
            "tags": ["health", "fitness", "exercise", "home"],
            "timestamp": (datetime.now() - timedelta(days=6)).isoformat(),
            "embedding": np.random.normal(0.05, 0.1, 384).tolist()
        },
        
        # Entertainment memories
        {
            "id": "mem_009",
            "content": "Đề xuất phim hay để xem cuối tuần",
            "context": "Giải trí và thư giãn",
            "importance_score": 0.45,
            "access_count": 3,
            "tags": ["entertainment", "movies", "recommendations"],
            "timestamp": (datetime.now() - timedelta(days=8)).isoformat(),
            "embedding": np.random.normal(-0.1, 0.1, 384).tolist()
        },
        {
            "id": "mem_010",
            "content": "Nhạc Việt Nam hay để nghe khi làm việc",
            "context": "Âm nhạc và productivity",
            "importance_score": 0.50,
            "access_count": 5,
            "tags": ["entertainment", "music", "vietnamese", "work"],
            "timestamp": (datetime.now() - timedelta(days=9)).isoformat(),
            "embedding": np.random.normal(-0.05, 0.1, 384).tolist()
        }
    ]
    
    return demo_memories

def save_demo_data(filename="demo_memories.json"):
    """Lưu demo data ra file JSON"""
    demo_data = create_demo_data()
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Demo data saved to {filename}")
    print(f"📊 Created {len(demo_data)} sample memories")
    
    return filename

def launch_visualizer():
    """Launch Memory Visualizer app"""
    try:
        # Try to run visualizer
        visualizer_path = Path(__file__).parent / 'memory_visualizer.py'
        
        if not visualizer_path.exists():
            print(f"❌ Visualizer not found at: {visualizer_path}")
            return False
        
        print("🚀 Launching Memory Visualizer...")
        print("📝 Instructions:")
        print("   1. Select 'Upload JSON' từ sidebar")
        print("   2. Upload file demo_memories.json")
        print("   3. Explore các tabs để xem visualizations")
        print()
        
        subprocess.run([
            'streamlit', 'run', str(visualizer_path),
            '--server.headless', 'false'
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped.")
        return True
    except Exception as e:
        print(f"❌ Failed to launch visualizer: {e}")
        return False

def main():
    """Main demo function"""
    print("🧠 Memory Visualizer Demo")
    print("=" * 40)
    
    # Create demo data
    print("📊 Creating demo data...")
    demo_file = save_demo_data()
    
    # Show data info
    with open(demo_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📈 Demo Data Overview:")
    print(f"   - Total memories: {len(data)}")
    
    # Count tags
    all_tags = []
    for mem in data:
        all_tags.extend(mem['tags'])
    unique_tags = set(all_tags)
    print(f"   - Unique tags: {len(unique_tags)}")
    print(f"   - Tags: {', '.join(sorted(unique_tags))}")
    
    # Importance distribution
    importance_scores = [mem['importance_score'] for mem in data]
    print(f"   - Avg importance: {np.mean(importance_scores):.2f}")
    print(f"   - Importance range: {min(importance_scores):.2f} - {max(importance_scores):.2f}")
    
    print(f"\n💾 Demo file: {demo_file}")
    print("🎯 Ready to launch visualizer!")
    
    # Ask user if they want to launch
    response = input("\n🚀 Launch Memory Visualizer now? (y/n): ").lower().strip()
    
    if response == 'y':
        launch_visualizer()
    else:
        print(f"📝 Demo data ready in: {demo_file}")
        print("🔧 To launch manually:")
        print("   python run_visualizer.py")
        print("   # or #")
        print("   streamlit run memory_visualizer.py")

if __name__ == "__main__":
    main()
