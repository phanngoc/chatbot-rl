#!/usr/bin/env python3
"""
MANN CLI Chatbot
Command-line interface chatbot với Memory-Augmented Neural Network
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from pathlib import Path
import torch

# Add standalone_mann to path
sys.path.insert(0, str(Path(__file__).parent / "standalone_mann"))

from standalone_mann.mann_core import MemoryAugmentedNetwork, MemoryBankEntry
from standalone_mann.mann_config import MANNConfig
from standalone_mann.mann_api import MANNClient
from standalone_mann.mann_monitoring import MANNMonitor, PagerSystem, HealthChecker


class MANNChatbot:
    """CLI Chatbot với MANN integration"""
    
    def __init__(self, config: MANNConfig = None):
        self.config = config or MANNConfig()
        self.logger = self._setup_logging()
        
        # Initialize MANN model
        self.mann_model = None
        self.monitor = None
        self.pager = None
        self.health_checker = None
        
        # Chat session
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.user_preferences = {}
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_memories_created": 0,
            "total_memories_retrieved": 0,
            "session_start": datetime.now()
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(self.config.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("MANNChatbot")
    
    async def initialize(self):
        """Initialize MANN system"""
        try:
            self.logger.info("Initializing MANN Chatbot...")
            
            # Initialize MANN model
            self.mann_model = MemoryAugmentedNetwork(
                input_size=self.config.input_size,
                hidden_size=self.config.hidden_size,
                memory_size=self.config.memory_size,
                memory_dim=self.config.memory_dim,
                output_size=self.config.output_size,
                device=torch.device('cpu')
            )
            
            # Load existing memory bank
            if os.path.exists(self.config.memory_save_path):
                self.mann_model.load_memory_bank(self.config.memory_save_path)
                self.logger.info(f"Loaded existing memory bank from {self.config.memory_save_path}")
            
            # Initialize monitoring
            if self.config.enable_monitoring:
                self.monitor = MANNMonitor(self.config)
                self.monitor.start_monitoring()
                
                if self.config.enable_pager:
                    self.pager = PagerSystem(self.config.pager_webhook_url)
                
                self.health_checker = HealthChecker(self.mann_model, self.monitor)
            
            self.logger.info("MANN Chatbot initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MANN Chatbot: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown MANN system"""
        try:
            self.logger.info("Shutting down MANN Chatbot...")
            
            # Save memory bank
            if self.mann_model:
                self.mann_model.save_memory_bank(self.config.memory_save_path)
                self.logger.info("Memory bank saved")
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
            
            self.logger.info("MANN Chatbot shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input và generate response"""
        try:
            self.stats["total_queries"] += 1
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "user_input": user_input,
                "session_id": self.session_id
            })
            
            # Convert input to tensor (simplified - in production use proper embeddings)
            # LSTM expects [batch_size, seq_len, input_size]
            input_tensor = torch.randn(1, 1, self.config.input_size)
            
            # Process with MANN
            start_time = datetime.now()
            try:
                output, memory_info = self.mann_model.forward(input_tensor, retrieve_memories=True)
            except Exception as e:
                self.logger.error(f"MANN forward error: {e}")
                raise e
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.stats["total_memories_retrieved"] += len(memory_info)
            
            # Record monitoring metrics
            if self.monitor:
                self.monitor.record_query(processing_time, len(memory_info))
            
            # Generate response based on retrieved memories
            response = await self._generate_response(user_input, memory_info)
            
            # Check if we should store this interaction as memory
            if self._should_store_memory(user_input, response):
                await self._store_interaction_memory(user_input, response, memory_info)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            if self.monitor:
                self.monitor.record_error()
            return f"Xin lỗi, tôi gặp lỗi khi xử lý yêu cầu của bạn: {str(e)}"
    
    async def _generate_response(self, user_input: str, memory_info: List[Dict]) -> str:
        """Generate response based on user input và retrieved memories"""
        
        # Try to use memory_info first, but fallback to search if needed
        relevant_memories = []
        
        # Check if memory_info contains actual memory content (not just "Memory slot X")
        if memory_info and any("Memory slot" not in mem.get("content", "") for mem in memory_info):
            # Use memory_info if it contains real content
            relevant_memories = [mem.get("content", "") for mem in memory_info if mem.get("content") and "Memory slot" not in mem.get("content", "")]
        elif self.mann_model.memory_bank:
            # Fallback to search from memory bank
            search_results = self.mann_model.search_memories(user_input, top_k=3)
            relevant_memories = [mem["content"] for mem in search_results]
        
        # Simple response generation (in production, use LLM)
        if not relevant_memories:
            return f"Tôi hiểu bạn đang nói về: '{user_input}'. Đây là lần đầu tiên tôi nghe về chủ đề này. Bạn có thể chia sẻ thêm thông tin không?"
        
        # Use retrieved memories to generate response
        memory_context = " ".join(relevant_memories[:1])  # Use top 1 memory only
        
        # Simple template-based response
        if "tên" in user_input.lower() or "name" in user_input.lower():
            return f"Tôi nhớ bạn đã nói về tên. Bạn có thể nhắc lại tên của bạn không?"
        elif "thích" in user_input.lower() or "like" in user_input.lower():
            return f"Tôi biết bạn có sở thích. Bạn có thể chia sẻ thêm về sở thích của bạn không?"
        elif "làm gì" in user_input.lower() or "what" in user_input.lower():
            return f"Bạn đang làm gì? Tôi quan tâm đến hoạt động của bạn."
        elif "nhớ" in user_input.lower() or "remember" in user_input.lower():
            return f"Tôi đang cố gắng nhớ những gì bạn đã nói. Bạn có thể nhắc lại không?"
        else:
            return f"Tôi hiểu bạn đang nói về: '{user_input}'. Bạn có thể chia sẻ thêm thông tin không?"
    
    def _should_store_memory(self, user_input: str, response: str) -> bool:
        """Determine if interaction should be stored as memory"""
        # Simple heuristics for memory storage
        important_keywords = ["tên", "name", "thích", "like", "làm", "do", "quan trọng", "important"]
        
        # Check if input contains important information
        user_lower = user_input.lower()
        has_important_keywords = any(keyword in user_lower for keyword in important_keywords)
        
        # Check input length (longer inputs are more likely to be important)
        is_substantial = len(user_input.split()) >= 3
        
        # Check if this is new information (not in recent conversation)
        recent_inputs = [conv["user_input"].lower() for conv in self.conversation_history[-5:]]
        is_new = not any(user_lower in recent for recent in recent_inputs)
        
        # Check if this content already exists in memory bank (prevent duplicates)
        content_to_check = f"User: {user_input}\nBot: {response}"
        existing_contents = [mem.content for mem in self.mann_model.memory_bank]
        is_duplicate = any(content_to_check in existing_content for existing_content in existing_contents)
        
        return (has_important_keywords or (is_substantial and is_new)) and not is_duplicate
    
    async def _store_interaction_memory(self, user_input: str, response: str, memory_info: List[Dict]):
        """Store interaction as memory"""
        try:
            # Extract key information
            content = f"User: {user_input}\nBot: {response}"
            context = f"Session: {self.session_id}, Retrieved memories: {len(memory_info)}"
            
            # Determine importance based on content
            importance = self._calculate_importance(user_input, response)
            
            # Add memory
            memory_id = self.mann_model.add_memory(
                content=content,
                context=context,
                tags=["user_interaction", f"session_{self.session_id}"],
                importance_weight=importance,
                metadata={
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "retrieved_memories": len(memory_info),
                    "user_input_length": len(user_input),
                    "response_length": len(response)
                }
            )
            
            self.stats["total_memories_created"] += 1
            self.logger.info(f"Stored interaction memory: {memory_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store interaction memory: {e}")
    
    def _calculate_importance(self, user_input: str, response: str) -> float:
        """Calculate importance weight for memory"""
        base_importance = 1.0
        
        # Increase importance for personal information
        personal_keywords = ["tên", "name", "tuổi", "age", "sống", "live", "làm việc", "work"]
        if any(keyword in user_input.lower() for keyword in personal_keywords):
            base_importance += 0.5
        
        # Increase importance for preferences
        preference_keywords = ["thích", "like", "yêu", "love", "ghét", "hate", "không thích", "dislike"]
        if any(keyword in user_input.lower() for keyword in preference_keywords):
            base_importance += 0.3
        
        # Increase importance for longer interactions
        if len(user_input.split()) > 10:
            base_importance += 0.2
        
        return min(base_importance, 3.0)  # Cap at 3.0
    
    async def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memories"""
        try:
            results = self.mann_model.search_memories(query, top_k=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            mann_stats = self.mann_model.get_memory_statistics()
            return {
                **mann_stats,
                **self.stats,
                "session_duration": (datetime.now() - self.stats["session_start"]).total_seconds(),
                "conversation_length": len(self.conversation_history)
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        if self.health_checker:
            return await self.health_checker.check_health()
        else:
            return {"status": "healthy", "message": "Health checker not available"}


class CLIManager:
    """Command-line interface manager"""
    
    def __init__(self, chatbot: MANNChatbot):
        self.chatbot = chatbot
        self.running = False
    
    async def start_interactive_mode(self):
        """Start interactive chat mode"""
        print("🤖 MANN Chatbot - Interactive Mode")
        print("=" * 50)
        print("Commands:")
        print("  /help     - Show help")
        print("  /stats    - Show statistics")
        print("  /search   - Search memories")
        print("  /health   - Health check")
        print("  /quit     - Exit")
        print("=" * 50)
        
        self.running = True
        
        while self.running:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input)
                else:
                    # Process normal input
                    response = await self.chatbot.process_user_input(user_input)
                    print(f"🤖 Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    async def _handle_command(self, command: str):
        """Handle CLI commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == '/help':
            self._show_help()
        elif cmd == '/stats':
            await self._show_statistics()
        elif cmd == '/search':
            if len(parts) > 1:
                query = ' '.join(parts[1:])
                await self._search_memories(query)
            else:
                print("Usage: /search <query>")
        elif cmd == '/health':
            await self._show_health()
        elif cmd == '/quit':
            self.running = False
            print("👋 Goodbye!")
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Available Commands:")
        print("  /help     - Show this help message")
        print("  /stats    - Show chatbot statistics")
        print("  /search   - Search memories (usage: /search <query>)")
        print("  /health   - Perform health check")
        print("  /quit     - Exit the chatbot")
    
    async def _show_statistics(self):
        """Show chatbot statistics"""
        stats = await self.chatbot.get_memory_statistics()
        print("\n📊 Chatbot Statistics:")
        print(f"  Total queries: {stats.get('total_queries', 0)}")
        print(f"  Total memories: {stats.get('total_memories', 0)}")
        print(f"  Memories created: {stats.get('total_memories_created', 0)}")
        print(f"  Memories retrieved: {stats.get('total_memories_retrieved', 0)}")
        print(f"  Memory utilization: {stats.get('memory_utilization', 0.0):.2%}")
        print(f"  Session duration: {stats.get('session_duration', 0):.1f} seconds")
        print(f"  Conversation length: {stats.get('conversation_length', 0)}")
    
    async def _search_memories(self, query: str):
        """Search memories"""
        print(f"\n🔍 Searching memories for: '{query}'")
        results = await self.chatbot.search_memories(query, top_k=5)
        
        if not results:
            print("  No memories found.")
            return
        
        print(f"  Found {len(results)} memories:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['content'][:100]}...")
            print(f"     Similarity: {result.get('similarity', 0):.3f}")
            print(f"     Importance: {result.get('importance_weight', 0):.2f}")
            print()
    
    async def _show_health(self):
        """Show health status"""
        health = await self.chatbot.health_check()
        print("\n🏥 Health Check:")
        print(f"  Status: {health.get('status', 'unknown')}")
        
        checks = health.get('checks', {})
        for check_name, check_data in checks.items():
            status = check_data.get('status', 'unknown')
            status_emoji = "✅" if status == "healthy" else "❌"
            print(f"  {status_emoji} {check_name}: {status}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MANN CLI Chatbot")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--api-mode", action="store_true", help="Run in API mode")
    parser.add_argument("--api-host", type=str, default="localhost", help="API host")
    parser.add_argument("--api-port", type=int, default=8000, help="API port")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Load configuration
    config = MANNConfig()
    if args.config:
        config = MANNConfig.from_file(args.config)
    
    # Override with command line arguments
    if args.log_level:
        config.log_level = args.log_level
    if args.api_host:
        config.api_host = args.api_host
    if args.api_port:
        config.api_port = args.api_port
    
    # Update from environment
    config.update_from_env()
    
    # Initialize chatbot
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        if args.api_mode:
            # Run API server
            from standalone_mann.mann_api import run_server
            print(f"🚀 Starting MANN API server on {config.api_host}:{config.api_port}")
            run_server(config)
        else:
            # Run CLI mode
            cli_manager = CLIManager(chatbot)
            await cli_manager.start_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await chatbot.shutdown()


if __name__ == "__main__":
    import torch
    asyncio.run(main())
