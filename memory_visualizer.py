"""
Memory Visualization App với Streamlit
Visualize cả vector representations và knowledge graph của memory system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import sys
import os

# Add src to path để import local modules
sys.path.append('src')

# Import optional dependencies
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

try:
    from memory.memory_manager import IntelligentMemoryManager, LLMExtractor
    from memory.retrieval_memory import RetrievalAugmentedMemory, EpisodicMemory
    from memory.consolidation import MemoryConsolidationSystem
    MEMORY_MODULES_AVAILABLE = True
except ImportError as e:
    MEMORY_MODULES_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Memory Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MemoryVisualizer:
    """Class chính để visualize memory system"""
    
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if 'memory_system' not in st.session_state:
            st.session_state.memory_system = None
        if 'consolidation_system' not in st.session_state:
            st.session_state.consolidation_system = None
        if 'memories_data' not in st.session_state:
            st.session_state.memories_data = []
        if 'graph_data' not in st.session_state:
            st.session_state.graph_data = None
            
    def load_memory_systems(self):
        """Load memory systems để visualization"""
        if not MEMORY_MODULES_AVAILABLE:
            st.error("❌ Memory modules not available. Cannot load memory systems.")
            return False
            
        try:
            # Initialize memory system
            if st.session_state.memory_system is None:
                memory_system = RetrievalAugmentedMemory(store_type="chroma")
                st.session_state.memory_system = memory_system
                
            # Initialize consolidation system
            if st.session_state.consolidation_system is None:
                consolidation_system = MemoryConsolidationSystem()
                st.session_state.consolidation_system = consolidation_system
                
            return True
        except Exception as e:
            st.error(f"Lỗi khi load memory systems: {e}")
            return False
    
    def load_memories_from_store(self):
        """Load memories từ memory store"""
        try:
            memory_system = st.session_state.memory_system
            if memory_system and hasattr(memory_system.store, 'memories'):
                memories = []
                for memory_id, memory in memory_system.store.memories.items():
                    memory_data = {
                        'id': memory.id,
                        'content': memory.content,
                        'context': memory.context,
                        'importance_score': memory.importance_score,
                        'access_count': memory.access_count,
                        'tags': memory.tags,
                        'timestamp': memory.timestamp,
                        'embedding': memory.embedding
                    }
                    memories.append(memory_data)
                
                st.session_state.memories_data = memories
                return memories
            return []
        except Exception as e:
            st.error(f"Lỗi khi load memories: {e}")
            return []
    
    def create_sample_memories(self, count: int = 50):
        """Tạo sample memories để demo"""
        sample_memories = []
        sample_texts = [
            "User hỏi về cách nấu phở",
            "Người dùng quan tâm đến machine learning",
            "Thảo luận về du lịch Việt Nam",
            "Hỏi về cách học tiếng Anh",
            "Quan tâm đến đầu tư chứng khoán",
            "Thích nghe nhạc pop",
            "Muốn tìm hiểu về AI",
            "Hỏi về công thức làm bánh",
            "Thảo luận về thể thao",
            "Quan tâm đến sức khỏe",
        ]
        
        contexts = [
            "Cuộc trò chuyện về ẩm thực",
            "Thảo luận công nghệ",
            "Chia sẻ kinh nghiệm du lịch",
            "Học tập và giáo dục",
            "Tài chính cá nhân",
            "Giải trí và âm nhạc",
            "Khoa học và công nghệ",
            "Nấu ăn và ẩm thực",
            "Thể thao và sức khỏe",
            "Sức khỏe và y tế"
        ]
        
        for i in range(count):
            text_idx = i % len(sample_texts)
            context_idx = i % len(contexts)
            
            # Generate fake embedding
            embedding = np.random.normal(0, 1, 384).tolist()
            
            memory_data = {
                'id': f'memory_{i}',
                'content': f"{sample_texts[text_idx]} - Sample {i}",
                'context': contexts[context_idx],
                'importance_score': np.random.uniform(0.1, 1.0),
                'access_count': np.random.randint(0, 20),
                'tags': [f'tag_{text_idx}', 'sample'],
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'embedding': embedding
            }
            sample_memories.append(memory_data)
        
        return sample_memories
    
    def visualize_vector_space(self, memories: List[Dict], method: str = "tsne"):
        """Visualize memory vectors trong 2D space"""
        if not memories:
            st.warning("Không có memories để visualize")
            return
        
        # Extract embeddings
        embeddings = []
        labels = []
        colors = []
        sizes = []
        
        for memory in memories:
            if memory.get('embedding'):
                embeddings.append(memory['embedding'])
                labels.append(f"{memory['content'][:50]}...")
                colors.append(memory['importance_score'])
                sizes.append(memory['access_count'] + 5)  # Min size 5
        
        if not embeddings:
            st.warning("Không có embeddings để visualize")
            return
        
        embeddings = np.array(embeddings)
        
        # Dimension reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == "umap":
            if not UMAP_AVAILABLE:
                st.error("❌ UMAP not available. Please install umap-learn.")
                return None
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)
        
        coords_2d = reducer.fit_transform(embeddings)
        
        # Create plotly figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="Importance Score"),
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><br>Importance: %{marker.color:.2f}<br>Access Count: %{marker.size}<extra></extra>',
            name='Memories'
        ))
        
        fig.update_layout(
            title=f'Memory Vector Space - {method.upper()}',
            xaxis_title=f'{method.upper()} Dimension 1',
            yaxis_title=f'{method.upper()} Dimension 2',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def visualize_knowledge_graph(self, consolidation_system):
        """Visualize knowledge graph"""
        try:
            graph = consolidation_system.knowledge_graph.graph
            
            if graph.number_of_nodes() == 0:
                st.warning("Knowledge graph trống")
                return None
            
            # Create layout
            pos = nx.spring_layout(graph, k=1, iterations=50)
            
            # Prepare node traces
            node_trace = go.Scatter(
                x=[], y=[], mode='markers+text',
                marker=dict(size=[], color=[], colorscale='Blues', 
                          line=dict(width=2, color='white')),
                text=[], textposition="middle center",
                textfont=dict(size=10),
                hovertemplate='<b>%{text}</b><br>Type: %{customdata}<extra></extra>',
                customdata=[]
            )
            
            # Prepare edge traces
            edge_traces = []
            
            for node in graph.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                
                node_info = graph.nodes[node]
                node_type = node_info.get('type', 'unknown')
                
                if node_type == 'concept':
                    node_trace['text'] += tuple([node_info.get('name', node)[:15]])
                    node_trace['marker']['size'] += tuple([20])
                    node_trace['marker']['color'] += tuple([1])
                else:  # knowledge
                    node_trace['text'] += tuple([f"K-{node[:10]}"])
                    node_trace['marker']['size'] += tuple([30])
                    node_trace['marker']['color'] += tuple([2])
                
                node_trace['customdata'] += tuple([node_type])
            
            # Add edges
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
            
            # Create figure
            fig = go.Figure(data=[node_trace] + edge_traces)
            
            fig.update_layout(
                title='Knowledge Graph Visualization',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Blue = Concepts, Darker Blue = Knowledge Nodes",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white',
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Lỗi khi visualize knowledge graph: {e}")
            return None
    
    def create_memory_metrics_dashboard(self, memories: List[Dict]):
        """Tạo dashboard với memory metrics"""
        if not memories:
            return
        
        # Calculate metrics
        total_memories = len(memories)
        avg_importance = np.mean([m['importance_score'] for m in memories])
        avg_access = np.mean([m['access_count'] for m in memories])
        
        # Time distribution
        timestamps = [m['timestamp'] for m in memories]
        df_time = pd.DataFrame({'timestamp': timestamps})
        df_time['date'] = pd.to_datetime(df_time['timestamp']).dt.date
        time_counts = df_time['date'].value_counts().sort_index()
        
        # Tags distribution
        all_tags = []
        for m in memories:
            all_tags.extend(m.get('tags', []))
        tag_counts = pd.Series(all_tags).value_counts().head(10)
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Memories", total_memories)
        with col2:
            st.metric("Avg Importance", f"{avg_importance:.2f}")
        with col3:
            st.metric("Avg Access Count", f"{avg_access:.1f}")
        with col4:
            unique_tags = len(set(all_tags))
            st.metric("Unique Tags", unique_tags)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Time distribution
            fig_time = px.line(
                x=time_counts.index, 
                y=time_counts.values,
                title="Memory Creation Over Time",
                labels={'x': 'Date', 'y': 'Number of Memories'}
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            # Tag distribution
            fig_tags = px.bar(
                x=tag_counts.index,
                y=tag_counts.values,
                title="Top 10 Tags",
                labels={'x': 'Tags', 'y': 'Count'}
            )
            fig_tags.update_layout(xaxis={'tickangle': 45})
            st.plotly_chart(fig_tags, use_container_width=True)
        
        # Importance vs Access scatter
        fig_scatter = px.scatter(
            x=[m['importance_score'] for m in memories],
            y=[m['access_count'] for m in memories],
            title="Importance Score vs Access Count",
            labels={'x': 'Importance Score', 'y': 'Access Count'},
            hover_data={'content': [m['content'][:50] + '...' for m in memories]}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    def run(self):
        """Main function để chạy Streamlit app"""
        st.title("🧠 Memory Visualization Dashboard")
        st.markdown("Visualize memory vectors và knowledge graph của chatbot RL system")
        
        # Show warnings for missing dependencies
        if not MEMORY_MODULES_AVAILABLE:
            st.error("❌ Memory modules not available. Real memory store features will be disabled.")
            st.info("💡 You can still use Sample Data và Upload JSON features.")
        
        if not UMAP_AVAILABLE:
            st.warning("⚠️ UMAP not installed. Only t-SNE và PCA will be available for vector visualization.")
        
        # Sidebar controls
        st.sidebar.header("⚙️ Controls")
        
        # Load systems
        if st.sidebar.button("🔄 Load Memory Systems"):
            with st.spinner("Loading memory systems..."):
                success = self.load_memory_systems()
                if success:
                    st.sidebar.success("✅ Memory systems loaded!")
                else:
                    st.sidebar.error("❌ Failed to load systems")
        
        # Data source selection
        data_source = st.sidebar.selectbox(
            "📊 Data Source",
            ["Sample Data", "Real Memory Store", "Upload JSON"]
        )
        
        memories_data = []
        
        if data_source == "Sample Data":
            num_samples = st.sidebar.slider("Number of samples", 10, 200, 50)
            if st.sidebar.button("🎲 Generate Sample Data"):
                memories_data = self.create_sample_memories(num_samples)
                st.sidebar.success(f"✅ Generated {len(memories_data)} sample memories")
        
        elif data_source == "Real Memory Store":
            if st.sidebar.button("📥 Load from Memory Store"):
                memories_data = self.load_memories_from_store()
                if memories_data:
                    st.sidebar.success(f"✅ Loaded {len(memories_data)} memories")
                else:
                    st.sidebar.warning("⚠️ No memories found in store")
        
        elif data_source == "Upload JSON":
            uploaded_file = st.sidebar.file_uploader("Choose JSON file", type="json")
            if uploaded_file is not None:
                try:
                    memories_data = json.load(uploaded_file)
                    st.sidebar.success(f"✅ Uploaded {len(memories_data)} memories")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading file: {e}")
        
        # Main content tabs
        if memories_data or st.session_state.memories_data:
            if memories_data:
                st.session_state.memories_data = memories_data
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Overview", 
                "🎯 Vector Space", 
                "🕸️ Knowledge Graph", 
                "🔍 Memory Explorer"
            ])
            
            with tab1:
                st.header("📊 Memory Metrics Dashboard")
                self.create_memory_metrics_dashboard(st.session_state.memories_data)
            
            with tab2:
                st.header("🎯 Vector Space Visualization")
                
                # Method selection
                col1, col2 = st.columns([1, 3])
                with col1:
                    available_methods = ["tsne", "pca"]
                    if UMAP_AVAILABLE:
                        available_methods.insert(1, "umap")
                    
                    method = st.selectbox(
                        "Reduction Method",
                        available_methods,
                        help="Method để reduce dimensions của vectors"
                    )
                    
                    if not UMAP_AVAILABLE and method == "umap":
                        st.warning("⚠️ UMAP not available. Install umap-learn để sử dụng.")
                
                if st.button("🎨 Generate Vector Visualization"):
                    with st.spinner(f"Generating {method.upper()} visualization..."):
                        fig = self.visualize_vector_space(st.session_state.memories_data, method)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.header("🕸️ Knowledge Graph Visualization")
                
                if st.session_state.consolidation_system:
                    # Check if graph has data
                    graph = st.session_state.consolidation_system.knowledge_graph.graph
                    
                    if graph.number_of_nodes() == 0:
                        st.info("📝 Knowledge graph trống. Hãy consolidate memories trước.")
                        
                        if st.button("🔄 Run Memory Consolidation"):
                            with st.spinner("Running memory consolidation..."):
                                try:
                                    # Convert memories to proper format for consolidation
                                    memories_for_consolidation = []
                                    for mem in st.session_state.memories_data:
                                        memories_for_consolidation.append({
                                            'id': mem['id'],
                                            'content': mem['content'],
                                            'context': mem['context'],
                                            'reward': mem['importance_score']  # Use importance as reward
                                        })
                                    
                                    results = st.session_state.consolidation_system.consolidate_memories(
                                        memories_for_consolidation, method="graph"
                                    )
                                    st.success(f"✅ Consolidation completed: {results}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Consolidation failed: {e}")
                    else:
                        st.info(f"📈 Graph có {graph.number_of_nodes()} nodes và {graph.number_of_edges()} edges")
                        
                        if st.button("🎨 Generate Graph Visualization"):
                            with st.spinner("Generating knowledge graph..."):
                                fig = self.visualize_knowledge_graph(st.session_state.consolidation_system)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Consolidation system chưa được load")
            
            with tab4:
                st.header("🔍 Memory Explorer")
                
                # Search memories
                search_query = st.text_input("🔍 Search memories:", placeholder="Enter search term...")
                
                if search_query:
                    # Simple search
                    filtered_memories = [
                        m for m in st.session_state.memories_data 
                        if search_query.lower() in m['content'].lower() or 
                           search_query.lower() in m['context'].lower()
                    ]
                    st.write(f"Found {len(filtered_memories)} matching memories")
                else:
                    filtered_memories = st.session_state.memories_data
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    min_importance = st.slider("Min Importance", 0.0, 1.0, 0.0)
                with col2:
                    min_access = st.slider("Min Access Count", 0, 20, 0)
                with col3:
                    # Tag filter
                    all_tags = []
                    for m in st.session_state.memories_data:
                        all_tags.extend(m.get('tags', []))
                    unique_tags = list(set(all_tags))
                    
                    selected_tags = st.multiselect("Filter by Tags", unique_tags)
                
                # Apply filters
                if min_importance > 0 or min_access > 0 or selected_tags:
                    filtered_memories = [
                        m for m in filtered_memories
                        if m['importance_score'] >= min_importance and
                           m['access_count'] >= min_access and
                           (not selected_tags or any(tag in m.get('tags', []) for tag in selected_tags))
                    ]
                
                # Display filtered memories
                st.write(f"📝 Showing {len(filtered_memories)} memories")
                
                for i, memory in enumerate(filtered_memories[:20]):  # Show first 20
                    with st.expander(f"Memory {i+1}: {memory['content'][:60]}..."):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Content:** {memory['content']}")
                            st.write(f"**Context:** {memory['context']}")
                            st.write(f"**Tags:** {', '.join(memory.get('tags', []))}")
                        
                        with col2:
                            st.metric("Importance", f"{memory['importance_score']:.3f}")
                            st.metric("Access Count", memory['access_count'])
                            st.write(f"**Timestamp:** {memory['timestamp']}")
                            st.write(f"**ID:** {memory['id']}")
        
        else:
            st.info("👆 Hãy chọn data source và load data để bắt đầu visualization")
            
            # Show instructions
            st.markdown("""
            ## 📖 Hướng dẫn sử dụng
            
            1. **Load Memory Systems**: Click để khởi tạo memory và consolidation systems
            2. **Chọn Data Source**: 
               - **Sample Data**: Tạo dữ liệu mẫu để demo
               - **Real Memory Store**: Load từ ChromaDB store thực tế
               - **Upload JSON**: Upload file JSON chứa memory data
            3. **Explore các tabs**:
               - **Overview**: Xem metrics và statistics tổng quan
               - **Vector Space**: Visualize memory vectors trong 2D space
               - **Knowledge Graph**: Xem graph relationships giữa concepts
               - **Memory Explorer**: Search và filter memories chi tiết
            
            ## 🔧 Yêu cầu
            - ChromaDB để vector storage
            - OpenAI API key (optional, cho consolidation)
            - Memory data với embeddings
            """)

# Run the app
if __name__ == "__main__":
    visualizer = MemoryVisualizer()
    visualizer.run()
