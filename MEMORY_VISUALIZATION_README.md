# 🧠 Memory Visualization Dashboard

Ứng dụng Streamlit để visualize cả **vector representations** và **knowledge graph** của memory system trong chatbot RL.

## ✨ Tính năng chính

### 📊 Overview Dashboard
- **Memory Metrics**: Tổng số memories, importance scores, access counts
- **Time Distribution**: Chart phân bố memory theo thời gian
- **Tag Analysis**: Top tags và distribution
- **Scatter Plot**: Quan hệ giữa importance và access count

### 🎯 Vector Space Visualization
- **t-SNE**: Non-linear dimensionality reduction cho clustering
- **UMAP**: Uniform Manifold Approximation cho structure preservation  
- **PCA**: Linear reduction cho quick overview
- **Interactive**: Hover để xem chi tiết memory, color-coded importance

### 🕸️ Knowledge Graph
- **Node Types**: Concepts (blue) vs Knowledge nodes (dark blue)
- **Relationships**: Edges show concept-knowledge connections
- **Interactive Layout**: Spring layout với hover details
- **Auto-consolidation**: Tự động consolidate memories thành graph

### 🔍 Memory Explorer
- **Search**: Full-text search trong content và context
- **Filters**: Filter theo importance, access count, tags
- **Detail View**: Expandable memory details
- **Real-time**: Live filtering và search

## 🚀 Cách sử dụng

### 1. Quick Start
```bash
# Chạy visualizer với script launcher
python run_visualizer.py

# Hoặc trực tiếp với Streamlit
streamlit run memory_visualizer.py
```

### 2. Data Sources

#### 🎲 Sample Data
- Tạo 10-200 sample memories
- Bao gồm embeddings, tags, timestamps
- Tốt cho demo và testing

#### 📥 Real Memory Store
- Load từ ChromaDB store thực tế
- Requires existing memory data
- Automatic embedding extraction

#### 📄 Upload JSON
- Upload file JSON với memory data
- Format: list of memory objects
- Flexible data import

### 3. Memory Data Format

```json
[
  {
    "id": "memory_1",
    "content": "User hỏi về cách nấu phở",
    "context": "Cuộc trò chuyện về ẩm thực",
    "importance_score": 0.85,
    "access_count": 5,
    "tags": ["food", "cooking"],
    "timestamp": "2024-01-15T10:30:00",
    "embedding": [0.1, 0.2, ...]  // 384-dim vector
  }
]
```

## 📋 Requirements

### Core Dependencies
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
umap-learn>=0.5.3
networkx>=2.8.0
```

### Memory System Dependencies
```
sentence-transformers>=2.2.0
chromadb>=0.4.0
faiss-cpu>=1.7.0
openai>=1.0.0  # Optional, for consolidation
```

## 🔧 Configuration

### Environment Variables
```bash
# ChromaDB path (optional)
export CHROMA_DB_PATH="src/data/chroma_db"

# OpenAI API key (for consolidation features)
export OPENAI_API_KEY="your-api-key"
```

### Script Options
```bash
# Custom port
python run_visualizer.py --port 8502

# No auto-open browser
python run_visualizer.py --no-auto-open

# Skip dependency check
python run_visualizer.py --skip-deps
```

## 📈 Visualization Methods

### Vector Space Reduction
- **t-SNE**: Best for discovering clusters và patterns
- **UMAP**: Good balance between local và global structure
- **PCA**: Fast linear reduction, good for overview

### Graph Layout
- **Spring Layout**: Force-directed layout cho natural clustering
- **Interactive**: Zoom, pan, hover for details
- **Color Coding**: Node type và importance visualization

## 🎨 Customization

### Theme và Styling
- Plotly white theme
- Color scales: Viridis for continuous, Blues for categorical
- Responsive layout với wide mode

### Performance
- Batch processing cho large datasets
- Caching for expensive computations
- Progressive loading for better UX

## 🐛 Troubleshooting

### Common Issues

1. **No memories found**
   - Check ChromaDB path
   - Verify memory data exists
   - Try sample data first

2. **Vector visualization empty**
   - Ensure memories have embeddings
   - Check embedding dimensions
   - Try different reduction method

3. **Graph visualization empty**
   - Run memory consolidation first
   - Check if concepts were extracted
   - Verify graph has nodes/edges

4. **Dependencies missing**
   - Run `python run_visualizer.py` for auto-install
   - Or manually: `pip install -r requirements.txt`

### Performance Tips
- Use sample data for large datasets (>1000 memories)
- Clear browser cache if visualization laggy
- Use PCA for quick preview before t-SNE/UMAP

## 📊 Example Workflows

### 1. Quick Demo
1. Launch app: `python run_visualizer.py`
2. Select "Sample Data" 
3. Generate 50 samples
4. Explore Overview tab
5. Try Vector Space với t-SNE

### 2. Real Data Analysis
1. Load memory systems
2. Select "Real Memory Store"
3. Load from ChromaDB
4. Analyze patterns in Vector Space
5. Run consolidation cho Knowledge Graph

### 3. Custom Data Import
1. Prepare JSON file với memory format
2. Upload via "Upload JSON"
3. Filter và search trong Memory Explorer
4. Export insights từ visualizations

## 🤝 Integration

### With Existing System
```python
# In your chatbot code
from memory_visualizer import MemoryVisualizer

visualizer = MemoryVisualizer()
visualizer.load_memory_systems()

# Export data for visualization
memories = visualizer.load_memories_from_store()
```

### API Endpoints (Future)
- REST API cho memory data
- Real-time updates
- Batch export/import

## 📝 Notes

- **Memory Requirements**: ~100MB RAM cho 1000 memories với embeddings
- **Browser**: Chrome/Firefox recommended cho best performance
- **Data Privacy**: All processing local, no external calls except OpenAI (optional)
- **Scalability**: Tested với up to 5000 memories

## 🔮 Future Enhancements

- [ ] Real-time memory updates
- [ ] Export visualizations as images
- [ ] Advanced graph algorithms (PageRank, community detection)
- [ ] Memory similarity heatmaps
- [ ] Temporal analysis và memory evolution
- [ ] Integration với training metrics
