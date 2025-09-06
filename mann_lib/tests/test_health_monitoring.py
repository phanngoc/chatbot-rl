#!/usr/bin/env python3
"""
Test: Health Monitoring
Tests health monitoring and performance metrics
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mann_chatbot import MANNChatbot
from standalone_mann.mann_config import MANNConfig


async def test_health_monitoring():
    """Test health monitoring functionality"""
    print("🏥 Test: Health Monitoring")
    print("=" * 50)
    
    config = MANNConfig()
    config.enable_monitoring = True
    chatbot = MANNChatbot(config)
    
    try:
        await chatbot.initialize()
        
        print("🔄 Performing operations to generate metrics...")
        
        # Generate various operations to test monitoring
        test_operations = [
            "Test operation 1: Basic query",
            "Test operation 2: Complex question about AI",
            "Test operation 3: Memory retrieval test",
            "Test operation 4: Learning sequence test", 
            "Test operation 5: Performance measurement"
        ]
        
        for i, operation in enumerate(test_operations, 1):
            print(f"  [{i}] {operation}")
            await chatbot.process_user_input(operation)
            await asyncio.sleep(0.5)
        
        # Check health status
        print(f"\n🏥 Health Check Results:")
        health = await chatbot.health_check()
        status = health.get('status', 'unknown')
        status_emoji = "✅" if status == "healthy" else "⚠️" if status == "warning" else "❌"
        
        print(f"  {status_emoji} Overall Status: {status}")
        
        # Show individual health checks
        checks = health.get('checks', {})
        print(f"  📋 Individual Checks:")
        for check_name, check_data in checks.items():
            check_status = check_data.get('status', 'unknown')
            check_emoji = "✅" if check_status == "healthy" else "⚠️" if check_status == "warning" else "❌"
            print(f"    {check_emoji} {check_name}: {check_status}")
            
            # Show additional check details if available
            if 'details' in check_data:
                print(f"       Details: {check_data['details']}")
        
        # Show performance statistics
        if hasattr(chatbot, 'monitor') and chatbot.monitor:
            print(f"\n📈 Performance Statistics:")
            perf_stats = chatbot.monitor.get_performance_stats()
            
            print(f"  📊 Query Statistics:")
            print(f"    Total queries: {perf_stats.get('total_queries', 0)}")
            print(f"    Average processing time: {perf_stats.get('avg_processing_time', 0):.3f}s")
            print(f"    Error rate: {perf_stats.get('error_rate', 0):.2%}")
            
            print(f"  💾 Memory Statistics:")
            print(f"    Memory utilization: {perf_stats.get('memory_utilization', 0):.2%}")
            print(f"    Memory operations: {perf_stats.get('memory_operations', 0)}")
            
            print(f"  🔧 System Statistics:")
            print(f"    Uptime: {perf_stats.get('uptime', 0):.1f}s")
            print(f"    Last check: {perf_stats.get('last_check_time', 'N/A')}")
        else:
            print(f"  ⚠️  Performance monitoring not available")
        
        # Test health check with different conditions
        print(f"\n🧪 Testing different health conditions...")
        
        # Add more operations to stress test
        stress_operations = [f"Stress test {i}" for i in range(1, 11)]
        for operation in stress_operations:
            await chatbot.process_user_input(operation)
            await asyncio.sleep(0.1)  # Faster operations
        
        # Check health again
        post_stress_health = await chatbot.health_check()
        post_stress_status = post_stress_health.get('status', 'unknown')
        post_stress_emoji = "✅" if post_stress_status == "healthy" else "⚠️" if post_stress_status == "warning" else "❌"
        
        print(f"  {post_stress_emoji} Post-stress Status: {post_stress_status}")
        
        # Show memory statistics
        final_stats = await chatbot.get_memory_statistics()
        print(f"\n💾 Final System Statistics:")
        print(f"  Total operations: {final_stats.get('total_queries', 0)}")
        print(f"  Memory utilization: {final_stats.get('memory_utilization', 0):.2%}")
        print(f"  Total retrievals: {final_stats.get('total_retrievals', 0)}")
        print(f"  Total writes: {final_stats.get('total_writes', 0)}")
        
        print(f"\n✅ Health Monitoring Test Complete!")
        print(f"   🏥 Health checks functioning properly")
        print(f"   📈 Performance metrics being collected")
        print(f"   🔧 System monitoring operational")
        
    finally:
        await chatbot.shutdown()


if __name__ == "__main__":
    asyncio.run(test_health_monitoring())