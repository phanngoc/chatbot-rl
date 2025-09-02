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
import tiktoken
from dotenv import load_dotenv
import openai

from agents.rl_chatbot import RLChatbotAgent

# Load environment variables
load_dotenv()


@st.cache_resource
def load_agent():
    """Load RL Chatbot Agent với caching"""
    # Load configuration from environment variables
    config = {
        "model_name": os.getenv("RL_MODEL_NAME", "microsoft/DialoGPT-medium"),
        "experience_buffer_size": int(os.getenv("RL_EXPERIENCE_BUFFER_SIZE", "5000")),
        "max_memories": int(os.getenv("RL_MAX_MEMORIES", "2000")),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.8"))
    }
    
    # Configure OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        st.warning("⚠️ OPENAI_API_KEY không được thiết lập. Vui lòng tạo file .env với API key của bạn.")
    
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
    
    # Start session if needed
    if not st.session_state.conversation_id:
        # Use new session-based approach
        session_id = agent.start_session()
        st.session_state.conversation_id = session_id
        st.success(f"Bắt đầu phiên trò chuyện mới: {session_id}")
        
        # Show session info
        with st.expander("📊 Thông tin Session", expanded=False):
            session_summary = agent.get_session_summary()
            st.json(session_summary)
    
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
                            # Rating system 1-5 stars
                            feedback_key = f"feedback_{metadata['experience_id']}"
                            st.write("**Đánh giá phản hồi:**")
                            
                            # Create 5 star rating buttons
                            rating_cols = st.columns(5)
                            for i in range(1, 6):
                                with rating_cols[i-1]:
                                    if st.button(f"⭐{i}", key=f"{feedback_key}_star_{i}"):
                                        # Convert 1-5 scale to [-1, 1] scale for internal use
                                        # 1→-1, 2→-0.5, 3→0, 4→0.5, 5→1
                                        feedback_score = (i - 3) / 2.0
                                        agent.provide_feedback(metadata['experience_id'], feedback_score)
                                        st.success(f"Cảm ơn bạn đã đánh giá {i}/5 sao!")
    
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
    
    # Tab interface cho different memory types
    tab1, tab2, tab3, tab4 = st.tabs(["🔎 Search Memories", "🧠 Episodic Experiences", "📚 Memory Bank", "📝 Recent Experiences"])
    
    with tab1:
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
    
    with tab2:
        # Episodic Experiences từ Meta-Learning System
        st.subheader("🧠 Episodic Experiences")
        
        try:
            meta_stats = agent.meta_learning_system.get_system_statistics()
            experience_buffer = agent.meta_learning_system.experience_buffer
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tổng Experiences", len(experience_buffer))
            with col2:
                st.metric("Episodes Trained", meta_stats['meta_learning']['episodes_trained'])
            with col3:
                st.metric("Avg Adaptation Loss", f"{meta_stats['meta_learning']['avg_adaptation_loss']:.4f}")
            with col4:
                st.metric("Memory Utilization", f"{meta_stats['meta_learning']['memory_utilization']:.2f}")
            
            # Experience filtering
            col1, col2 = st.columns(2)
            with col1:
                experience_count = st.slider("Số lượng experiences hiển thị", 5, 50, 10)
            with col2:
                sort_by = st.selectbox("Sắp xếp theo", ["Mới nhất", "Reward cao nhất", "Reward thấp nhất"])
            
            if experience_buffer:
                # Sort experiences
                experiences = list(experience_buffer)
                if sort_by == "Reward cao nhất":
                    experiences.sort(key=lambda x: x.get('reward', 0), reverse=True)
                elif sort_by == "Reward thấp nhất":
                    experiences.sort(key=lambda x: x.get('reward', 0))
                # "Mới nhất" keeps original order (most recent last)
                
                # Limit to requested count
                shown_experiences = experiences[-experience_count:] if sort_by == "Mới nhất" else experiences[:experience_count]
                
                st.write(f"**Hiển thị {len(shown_experiences)} experiences:**")
                
                for i, exp in enumerate(shown_experiences):
                    reward = exp.get('reward', 0)
                    reward_color = "🟢" if reward > 0 else "🔴" if reward < 0 else "🔵"
                    
                    with st.expander(f"{reward_color} Experience {i+1} - Reward: {reward:.3f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Context:**")
                            context = exp.get('context', '')
                            st.text_area("", context, height=120, key=f"episodic_context_{i}", disabled=True)
                            
                            st.write("**Metrics:**")
                            st.write(f"• Reward: **{reward:.3f}**")
                            if exp.get('user_feedback'):
                                st.write(f"• User Feedback: **{exp['user_feedback']}**")
                        
                        with col2:
                            st.write("**Response:**")
                            response = exp.get('response', '')
                            st.text_area("", response, height=120, key=f"episodic_response_{i}", disabled=True)
                            
                            timestamp = exp.get('timestamp')
                            if timestamp:
                                if hasattr(timestamp, 'isoformat'):
                                    st.write(f"**Timestamp:** {timestamp.isoformat()}")
                                elif hasattr(timestamp, 'item'):  # Tensor timestamp
                                    st.write(f"**Timestep:** {timestamp.item()}")
                                else:
                                    st.write(f"**Timestamp:** {str(timestamp)}")
            else:
                st.info("Chưa có episodic experiences nào trong meta-learning system")
                
        except Exception as e:
            st.error(f"Lỗi khi load episodic experiences: {e}")
    
    with tab3:
        # Memory Bank từ Meta-Learning
        st.subheader("📚 Memory Bank Entries")
        
        try:
            # Search trong memory bank
            search_query_mb = st.text_input("Tìm kiếm trong Memory Bank:", key="mb_search")
            top_k = st.slider("Số lượng memories", 1, 20, 5)
            
            if search_query_mb:
                memories = agent.meta_learning_system.select_relevant_memories(
                    search_query_mb, top_k=top_k
                )
                
                if memories:
                    st.success(f"Tìm thấy {len(memories)} memory entries")
                    
                    for i, memory in enumerate(memories):
                        similarity = memory.get('similarity', 0)
                        importance = memory.get('importance_weight', 1.0)
                        usage = memory.get('usage_count', 0)
                        
                        # Color coding based on quality
                        if similarity > 0.8 and importance > 1.5:
                            quality_icon = "🔥"
                            quality_text = "High Quality"
                        elif similarity > 0.6:
                            quality_icon = "⭐"
                            quality_text = "Good Quality"
                        else:
                            quality_icon = "💡"
                            quality_text = "Low Quality"
                        
                        with st.expander(f"{quality_icon} Memory {i+1} - {quality_text} (Sim: {similarity:.3f})"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Similarity", f"{similarity:.3f}")
                            with col2:
                                st.metric("Importance", f"{importance:.2f}")
                            with col3:
                                st.metric("Usage Count", usage)
                            
                            st.write("**Memory Details:**")
                            st.write(f"• Memory Index: {memory.get('memory_index', 'Unknown')}")
                            
                            # Thêm metadata nếu có
                            if hasattr(agent.meta_learning_system.mann, 'memory_bank'):
                                memory_idx = memory.get('memory_index', -1)
                                if memory_idx != -1 and memory_idx < len(agent.meta_learning_system.mann.memory_bank):
                                    entry = agent.meta_learning_system.mann.memory_bank[memory_idx]
                                    st.write(f"• Last Accessed: {entry.last_accessed}")
                                    st.write(f"• Key Shape: {entry.key.shape}")
                                    st.write(f"• Value Shape: {entry.value.shape}")
                else:
                    st.info("Không tìm thấy memories nào với query này")
            else:
                st.info("Nhập query để tìm kiếm trong memory bank")
                
        except Exception as e:
            st.error(f"Lỗi khi truy cập memory bank: {e}")
    
    with tab4:
        # Recent experiences từ Experience Buffer
        st.subheader("📝 Recent Experience Buffer")
        
        if len(agent.experience_buffer.buffer) > 0:
            # Buffer statistics
            try:
                buffer_stats = agent.experience_buffer.get_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Experiences", buffer_stats.get('total_experiences', 0))
                with col2:
                    st.metric("Buffer Utilization", f"{buffer_stats.get('buffer_utilization', 0):.1f}%")
                with col3:
                    st.metric("Avg Reward", f"{buffer_stats.get('avg_reward', 0):.3f}")
                with col4:
                    st.metric("Conversations", buffer_stats.get('total_conversations', 0))
            except Exception as e:
                st.error(f"Error loading buffer statistics: {e}")
                # Fallback metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Experiences", len(agent.experience_buffer.buffer))
                with col2:
                    st.metric("Buffer Utilization", "N/A")
                with col3:
                    st.metric("Avg Reward", "N/A")
                with col4:
                    st.metric("Conversations", "N/A")
            
            # Experience filtering
            exp_count = st.slider("Số lượng experiences", 5, 30, 10, key="exp_buffer_count")
            
            recent_experiences = list(agent.experience_buffer.buffer)[-exp_count:]  # Last N
            
            st.write(f"**{len(recent_experiences)} experiences gần đây từ Experience Buffer:**")
            
            for i, exp in enumerate(reversed(recent_experiences)):
                reward = exp.reward
                reward_color = "🟢" if reward > 0 else "🔴" if reward < 0 else "🔵"
                
                with st.expander(f"{reward_color} Experience {i+1} - Reward: {reward:.3f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**State (Context):**")
                        st.text_area("", exp.state, height=100, key=f"exp_buf_state_{i}", disabled=True)
                        
                        st.write("**Metrics:**")
                        st.write(f"• Reward: **{reward:.3f}**")
                        st.write(f"• Conversation ID: {exp.conversation_id}")
                        if exp.user_feedback:
                            st.write(f"• User Feedback: **{exp.user_feedback}**")
                    
                    with col2:
                        st.write("**Action (Response):**")
                        st.text_area("", exp.action, height=100, key=f"exp_buf_action_{i}", disabled=True)
                        
                        st.write("**Timestamp:**", exp.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
                        if exp.next_state:
                            st.write("**Next State:**", exp.next_state[:100] + "..." if len(exp.next_state) > 100 else exp.next_state)
        else:
            st.info("Experience buffer rỗng")
    
    # Consolidated knowledge (moved outside tabs)
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


def session_management():
    """Session Management interface với database integration"""
    st.header("📚 Quản lý Session")
    
    agent = st.session_state.agent
    
    # Current session info
    if st.session_state.conversation_id:
        st.subheader("Session hiện tại")
        
        # Session summary
        try:
            session_summary = agent.get_session_summary()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Session ID", session_summary.get("session_id", "Unknown")[:8] + "...")
            with col2:
                st.metric("Tổng tin nhắn", session_summary.get("total_messages", 0))
            with col3:
                st.metric("Memory Bank Size", session_summary.get("memory_stats", {}).get("total_entries", 0))
            
            # Detailed session info
            with st.expander("Chi tiết Session", expanded=False):
                st.json(session_summary)
        
        except Exception as e:
            st.error(f"Lỗi khi lấy thông tin session: {e}")
        
        # Session actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Bắt đầu session mới"):
                session_id = agent.start_session()
                st.session_state.conversation_id = session_id
                st.session_state.conversation_history = []
                st.success(f"Đã tạo session mới: {session_id[:8]}...")
                st.rerun()
        
        with col2:
            if st.button("💾 Lưu Memory Bank"):
                if agent.force_save_memory():
                    st.success("Đã lưu memory bank!")
                else:
                    st.error("Lỗi khi lưu memory bank")
        
        with col3:
            if st.button("🗑️ Clear Memory"):
                agent.clear_current_session_memory()
                st.success("Đã xóa memory của session!")
    
    # Recent sessions
    st.subheader("Sessions gần đây")
    try:
        recent_sessions = agent.list_recent_sessions(10)
        
        if recent_sessions:
            for session in recent_sessions:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.text(f"ID: {session['session_id'][:8]}...")
                    
                    with col2:
                        st.text(f"Messages: {session['total_messages']}")
                    
                    with col3:
                        last_updated = session.get('last_updated')
                        if isinstance(last_updated, str):
                            last_updated = last_updated[:16]  # Truncate datetime
                        st.text(f"Updated: {last_updated}")
                    
                    with col4:
                        if st.button("Resume", key=f"resume_{session['session_id']}"):
                            if agent.resume_session(session['session_id']):
                                st.session_state.conversation_id = session['session_id']
                                st.success("Đã chuyển sang session!")
                                st.rerun()
                            else:
                                st.error("Không thể resume session")
        else:
            st.info("Chưa có sessions nào")
    
    except Exception as e:
        st.error(f"Lỗi khi load sessions: {e}")
    
    # Database statistics
    st.subheader("Thống kê Database")
    try:
        db_stats = agent.get_database_stats()
        
        if "error" not in db_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tổng Sessions", db_stats.get("total_sessions", 0))
            
            with col2:
                st.metric("Tổng Messages", db_stats.get("total_messages", 0))
            
            with col3:
                st.metric("Active Sessions (7 days)", db_stats.get("recent_active_sessions", 0))
            
            # Detailed stats
            with st.expander("Chi tiết Database", expanded=False):
                st.json(db_stats)
        else:
            st.warning("Database không khả dụng")
    
    except Exception as e:
        st.error(f"Lỗi database stats: {e}")
    
    # Episodic Experiences
    st.subheader("🧠 Episodic Experiences")
    try:
        # Get meta-learning system stats
        meta_stats = agent.meta_learning_system.get_system_statistics()
        experience_buffer = agent.meta_learning_system.experience_buffer
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng Experiences", len(experience_buffer))
        with col2:
            st.metric("Memory Bank Size", meta_stats['memory_bank']['total_memories'])
        with col3:
            st.metric("Timestep", meta_stats['memory_bank'].get('current_timestep', 0))
        
        # Display recent experiences
        if experience_buffer:
            st.write("**Experiences gần đây:**")
            
            # Limit to recent 10 experiences
            recent_experiences = experience_buffer[-10:] if len(experience_buffer) > 10 else experience_buffer
            
            for i, exp in enumerate(reversed(recent_experiences)):
                with st.expander(f"Experience {i+1} - Reward: {exp.get('reward', 0):.2f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Context:**")
                        st.text_area("", exp.get('context', ''), height=100, key=f"exp_context_{i}", disabled=True)
                        
                        st.write("**Reward:**", exp.get('reward', 0))
                        if exp.get('user_feedback'):
                            st.write("**User Feedback:**", exp.get('user_feedback'))
                    
                    with col2:
                        st.write("**Response:**")
                        st.text_area("", exp.get('response', ''), height=100, key=f"exp_response_{i}", disabled=True)
                        
                        timestamp = exp.get('timestamp')
                        if timestamp:
                            if hasattr(timestamp, 'isoformat'):
                                st.write("**Timestamp:**", timestamp.isoformat())
                            else:
                                st.write("**Timestamp:**", str(timestamp))
        else:
            st.info("Chưa có episodic experiences nào")
        
        # Memory Bank Information
        st.write("**Memory Bank Details:**")
        if meta_stats.get('memory_loaded_from_db'):
            st.success("✅ Memory bank đã được load từ database")
        else:
            st.warning("⚠️ Memory bank chưa được load từ database")
        
        # Memory Bank entries preview
        try:
            memories = agent.meta_learning_system.select_relevant_memories("sample query", top_k=5)
            if memories:
                st.write("**Sample Memory Entries:**")
                for i, memory in enumerate(memories):
                    with st.expander(f"Memory {i+1} - Similarity: {memory.get('similarity', 0):.3f}"):
                        st.write("**Memory Index:**", memory.get('memory_index', 'Unknown'))
                        st.write("**Importance Weight:**", memory.get('importance_weight', 1.0))
                        st.write("**Usage Count:**", memory.get('usage_count', 0))
        except Exception as e:
            st.warning(f"Không thể load memory entries: {e}")
    
    except Exception as e:
        st.error(f"Lỗi khi load episodic experiences: {e}")
    
    # Advanced actions
    st.subheader("Hành động nâng cao")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 Meta-learning Session"):
            with st.spinner("Running meta-learning..."):
                try:
                    results = agent.meta_learning_system.meta_learning_session(num_episodes=5)
                    st.json(results)
                except Exception as e:
                    st.error(f"Meta-learning failed: {e}")
    
    with col2:
        if st.button("🗑️ Cleanup Old Data"):
            with st.spinner("Cleaning up old data..."):
                try:
                    cleanup_results = agent.cleanup_old_data(days_threshold=30)
                    st.success(f"Cleaned up {cleanup_results['sessions_cleaned']} sessions")
                except Exception as e:
                    st.error(f"Cleanup failed: {e}")
    
    with col3:
        if st.button("📤 Export Current Session"):
            if st.session_state.conversation_id:
                output_path = f"data/session_export_{st.session_state.conversation_id[:8]}.json"
                try:
                    if agent.export_current_session(output_path):
                        st.success(f"Exported to: {output_path}")
                    else:
                        st.error("Export failed")
                except Exception as e:
                    st.error(f"Export error: {e}")
            else:
                st.warning("No active session to export")


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
            ["💬 Trò chuyện", "📚 Quản lý Session", "📊 Phân tích", "🔍 Khám phá bộ nhớ", "⚙️ Cài đặt"]
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
    elif page == "📚 Quản lý Session":
        session_management()
    elif page == "📊 Phân tích":
        analytics_dashboard()
    elif page == "🔍 Khám phá bộ nhớ":
        memory_explorer()
    elif page == "⚙️ Cài đặt":
        settings_page()


if __name__ == "__main__":
    main()
