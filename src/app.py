"""
Streamlit Web Interface cho RL Chatbot
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from agents.rl_chatbot import RLChatbotAgent


@st.cache_resource
def load_agent():
    """Load RL Chatbot Agent với caching"""
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "device": "cpu",
        "experience_buffer_size": 5000,
        "memory_store_type": "chroma",
        "max_memories": 2000,
        "consolidation_threshold": 50,
        "ewc_lambda": 500.0,
        "temperature": 0.8
    }
    
    agent = RLChatbotAgent(config=config)
    
    # Try to load existing state
    state_path = "data/agent_state.json"
    if os.path.exists(state_path):
        agent.load_agent_state(state_path)
    
    return agent


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    
    if 'agent' not in st.session_state:
        st.session_state.agent = load_agent()


def chat_interface():
    """Main chat interface"""
    st.header("🤖 RL Chatbot - Trò chuyện")
    
    agent = st.session_state.agent
    
    # Start conversation if needed
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = agent.start_conversation()
        st.success(f"Bắt đầu cuộc trò chuyện mới: {st.session_state.conversation_id}")
    
    # Chat input
    user_input = st.chat_input("Nhập tin nhắn của bạn...")
    
    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Process message
        with st.spinner("Đang suy nghĩ..."):
            result = agent.process_message(user_input)
        
        # Add bot response to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": result['response'],
            "timestamp": datetime.now(),
            "metadata": {
                "experience_id": result['experience_id'],
                "memories_used": result['relevant_memories_count'],
                "response_time_ms": result['response_time_ms']
            }
        })
    
    # Display conversation
    for i, message in enumerate(st.session_state.conversation_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show metadata for bot messages
                if "metadata" in message:
                    with st.expander("📊 Chi tiết"):
                        metadata = message["metadata"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Memories Used", metadata["memories_used"])
                        with col2:
                            st.metric("Response Time", f"{metadata['response_time_ms']:.1f}ms")
                        with col3:
                            # Feedback buttons
                            feedback_key = f"feedback_{metadata['experience_id']}"
                            
                            col_pos, col_neg = st.columns(2)
                            with col_pos:
                                if st.button("👍", key=f"{feedback_key}_pos"):
                                    agent.provide_feedback(metadata['experience_id'], 0.8)
                                    st.success("Cảm ơn feedback tích cực!")
                            with col_neg:
                                if st.button("👎", key=f"{feedback_key}_neg"):
                                    agent.provide_feedback(metadata['experience_id'], -0.8)
                                    st.success("Cảm ơn feedback, tôi sẽ cải thiện!")
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Điều khiển")
        
        if st.button("🔄 Cuộc trò chuyện mới"):
            st.session_state.conversation_history = []
            st.session_state.conversation_id = agent.start_conversation()
            st.rerun()
        
        if st.button("💾 Lưu trạng thái"):
            success = agent.save_agent_state("data/agent_state.json")
            if success:
                st.success("Đã lưu trạng thái!")
            else:
                st.error("Lỗi khi lưu trạng thái!")
        
        # Conversation stats
        if st.session_state.conversation_history:
            st.subheader("📊 Thống kê cuộc trò chuyện")
            
            user_messages = [m for m in st.session_state.conversation_history if m["role"] == "user"]
            bot_messages = [m for m in st.session_state.conversation_history if m["role"] == "assistant"]
            
            st.metric("Tổng tin nhắn", len(st.session_state.conversation_history))
            st.metric("Tin nhắn của bạn", len(user_messages))
            st.metric("Phản hồi của bot", len(bot_messages))
            
            if bot_messages:
                avg_response_time = sum(
                    m.get("metadata", {}).get("response_time_ms", 0) 
                    for m in bot_messages
                ) / len(bot_messages)
                st.metric("Thời gian phản hồi TB", f"{avg_response_time:.1f}ms")


def analytics_dashboard():
    """Analytics dashboard"""
    st.header("📊 Bảng điều khiển phân tích")
    
    agent = st.session_state.agent
    
    # Get system status
    status = agent.get_system_status()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tổng tương tác", 
            status['performance_metrics']['total_interactions']
        )
    
    with col2:
        st.metric(
            "Feedback tích cực", 
            status['performance_metrics']['positive_feedback']
        )
    
    with col3:
        st.metric(
            "Feedback tiêu cực", 
            status['performance_metrics']['negative_feedback']
        )
    
    with col4:
        st.metric(
            "Thời gian phản hồi TB", 
            f"{status['performance_metrics']['avg_response_time']:.2f}s"
        )
    
    # Memory systems status
    st.subheader("🧠 Trạng thái hệ thống bộ nhớ")
    
    memory_stats = status['memory_systems']
    
    # Experience buffer
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Experience Buffer")
        buffer_stats = memory_stats['experience_buffer']
        
        # Buffer utilization
        utilization = buffer_stats['buffer_utilization']
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = utilization,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Buffer Utilization (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Experience distribution
        if buffer_stats['total_experiences'] > 0:
            experience_data = {
                'Type': ['Positive', 'Negative', 'Neutral'],
                'Count': [
                    buffer_stats['positive_experiences'],
                    buffer_stats['negative_experiences'], 
                    buffer_stats['neutral_experiences']
                ]
            }
            
            fig = px.pie(
                values=experience_data['Count'],
                names=experience_data['Type'],
                title="Phân phối Experience"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Retrieval Memory")
        retrieval_stats = memory_stats['retrieval_memory']
        
        if retrieval_stats['total_memories'] > 0:
            st.metric("Tổng memories", retrieval_stats['total_memories'])
            st.metric("Avg importance", f"{retrieval_stats['avg_importance']:.2f}")
            st.metric("Max access count", retrieval_stats['max_access_count'])
            
            # Memory importance distribution
            importance_data = {
                'Category': ['Highly Important', 'Low Important', 'Others'],
                'Count': [
                    retrieval_stats['highly_important_memories'],
                    retrieval_stats['low_importance_memories'],
                    retrieval_stats['total_memories'] - 
                    retrieval_stats['highly_important_memories'] - 
                    retrieval_stats['low_importance_memories']
                ]
            }
            
            fig = px.bar(
                x=importance_data['Category'],
                y=importance_data['Count'],
                title="Phân phối Importance"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Chưa có memories nào")
    
    # Temporal weighting stats
    st.subheader("⏰ Temporal Weighting")
    temporal_stats = memory_stats['temporal_weighting']
    
    if temporal_stats['total_experiences'] > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Experiences", temporal_stats['total_experiences'])
        
        with col2:
            weight_dist = temporal_stats['weight_distribution']
            st.metric("Mean Weight", f"{weight_dist['mean']:.3f}")
        
        with col3:
            st.metric("High Weight Experiences", temporal_stats['high_weight_experiences'])
        
        # Weight components
        components = temporal_stats['component_weights']
        component_data = {
            'Component': ['Temporal', 'Importance', 'Access', 'Quality'],
            'Mean': [
                components['temporal']['mean'],
                components['importance']['mean'],
                components['access']['mean'],
                components['quality']['mean']
            ]
        }
        
        fig = px.bar(
            x=component_data['Component'],
            y=component_data['Mean'],
            title="Trung bình Weight Components"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Meta-learning stats
    st.subheader("🎯 Meta-Learning")
    meta_stats = memory_stats['meta_learning']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Episodes Trained", meta_stats['meta_learning']['episodes_trained'])
    
    with col2:
        st.metric("Experience Buffer Size", meta_stats['experience_buffer_size'])
    
    with col3:
        memory_bank = meta_stats['memory_bank']
        st.metric("Memory Utilization", f"{memory_bank['total_memories']}/1000")


def memory_explorer():
    """Memory exploration interface"""
    st.header("🔍 Khám phá bộ nhớ")
    
    agent = st.session_state.agent
    
    # Search interface
    st.subheader("🔎 Tìm kiếm memories")
    
    search_query = st.text_input("Nhập từ khóa tìm kiếm:")
    
    if search_query:
        # Search in retrieval memory
        memories = agent.retrieval_memory.retrieve_relevant_memories(
            search_query, top_k=10
        )
        
        if memories:
            st.success(f"Tìm thấy {len(memories)} memories liên quan")
            
            for i, memory in enumerate(memories):
                with st.expander(f"Memory {i+1} - Similarity: {memory['similarity']:.3f}"):
                    st.write("**Content:**", memory['content'])
                    st.write("**Context:**", memory['context'])
                    st.write("**Importance:**", memory['importance_score'])
                    st.write("**Access Count:**", memory['access_count'])
                    st.write("**Tags:**", ", ".join(memory['tags']))
                    
                    if memory['metadata']:
                        st.write("**Metadata:**", memory['metadata'])
        else:
            st.info("Không tìm thấy memories nào")
    
    # Recent experiences
    st.subheader("📝 Experiences gần đây")
    
    if len(agent.experience_buffer.buffer) > 0:
        recent_experiences = list(agent.experience_buffer.buffer)[-10:]  # Last 10
        
        for i, exp in enumerate(reversed(recent_experiences)):
            with st.expander(f"Experience {i+1} - Reward: {exp.reward:.2f}"):
                st.write("**State:**", exp.state)
                st.write("**Action:**", exp.action)
                st.write("**Reward:**", exp.reward)
                st.write("**Timestamp:**", exp.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                if exp.user_feedback:
                    st.write("**User Feedback:**", exp.user_feedback)
    else:
        st.info("Chưa có experiences nào")
    
    # Consolidated knowledge
    st.subheader("🧩 Consolidated Knowledge")
    
    consolidated_knowledge = agent.consolidation_system.consolidated_knowledge
    
    if consolidated_knowledge:
        st.success(f"Có {len(consolidated_knowledge)} knowledge items đã được consolidate")
        
        for knowledge_id, knowledge in list(consolidated_knowledge.items())[:5]:
            with st.expander(f"Knowledge: {knowledge.consolidation_method}"):
                st.write("**Summary:**", knowledge.summary)
                st.write("**Confidence:**", knowledge.confidence_score)
                st.write("**Source Memories:**", len(knowledge.source_memories))
                st.write("**Created:**", knowledge.created_at.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        st.info("Chưa có knowledge nào được consolidate")


def settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Cài đặt")
    
    agent = st.session_state.agent
    
    st.subheader("🔧 Cấu hình hệ thống")
    
    # Model settings
    with st.expander("🤖 Model Settings"):
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        if st.button("Cập nhật Temperature"):
            agent.config["temperature"] = temperature
            st.success("Đã cập nhật temperature!")
    
    # Memory settings
    with st.expander("🧠 Memory Settings"):
        consolidation_threshold = st.number_input(
            "Consolidation Threshold", 
            min_value=10, 
            max_value=1000, 
            value=100
        )
        
        ewc_lambda = st.number_input(
            "EWC Lambda", 
            min_value=100.0, 
            max_value=10000.0, 
            value=1000.0
        )
        
        if st.button("Cập nhật Memory Settings"):
            agent.consolidation_system.consolidation_threshold = consolidation_threshold
            agent.ewc_system.ewc_lambda = ewc_lambda
            st.success("Đã cập nhật memory settings!")
    
    # Data management
    st.subheader("📁 Quản lý dữ liệu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Xóa conversation history"):
            st.session_state.conversation_history = []
            st.session_state.conversation_id = None
            st.success("Đã xóa conversation history!")
    
    with col2:
        if st.button("🧹 Dọn dẹp memories cũ"):
            # Clean old experiences
            removed_count = agent.experience_buffer.clear_old_experiences(days_threshold=30)
            st.success(f"Đã xóa {removed_count} experiences cũ!")
    
    # Export/Import
    st.subheader("📤 Xuất/Nhập dữ liệu")
    
    if st.button("📤 Xuất agent state"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"data/agent_export_{timestamp}.json"
        success = agent.save_agent_state(export_path)
        
        if success:
            st.success(f"Đã xuất agent state tới: {export_path}")
        else:
            st.error("Lỗi khi xuất agent state!")
    
    # System info
    st.subheader("ℹ️ Thông tin hệ thống")
    
    status = agent.get_system_status()
    
    st.json({
        "Model Info": status['model_info'],
        "System Health": status['system_health']
    })


def main():
    st.set_page_config(
        page_title="RL Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🤖 RL Chatbot")
        st.markdown("---")
        
        page = st.selectbox(
            "Chọn trang:",
            ["💬 Trò chuyện", "📊 Phân tích", "🔍 Khám phá bộ nhớ", "⚙️ Cài đặt"]
        )
        
        st.markdown("---")
        
        # Quick stats
        agent = st.session_state.agent
        status = agent.get_system_status()
        
        st.metric("Total Interactions", status['performance_metrics']['total_interactions'])
        st.metric("Experience Buffer", f"{len(agent.experience_buffer.buffer)}/{agent.experience_buffer.max_size}")
    
    # Main content
    if page == "💬 Trò chuyện":
        chat_interface()
    elif page == "📊 Phân tích":
        analytics_dashboard()
    elif page == "🔍 Khám phá bộ nhớ":
        memory_explorer()
    elif page == "⚙️ Cài đặt":
        settings_page()


if __name__ == "__main__":
    main()
