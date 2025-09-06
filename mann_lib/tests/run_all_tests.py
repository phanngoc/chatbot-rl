#!/usr/bin/env python3
"""
Run All Tests
Executes all MANN test cases in sequence with proper reporting
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_basic_conversation import test_basic_conversation
from test_memory_search import test_memory_search  
from test_memory_management import test_memory_management
from test_external_working_memory import test_external_working_memory
from test_ppo_training import test_ppo_training
from test_ppo_importance_ratio import test_ppo_importance_ratio
from test_health_monitoring import test_health_monitoring
from test_api_integration import test_api_integration


async def run_all_tests():
    """Run all test cases in sequence"""
    print("🧪 MANN Test Suite")
    print("=" * 60)
    
    test_cases = [
        ("Basic Conversation", test_basic_conversation, True),
        ("Memory Search", test_memory_search, True),
        ("Memory Management", test_memory_management, True),
        ("External Working Memory", test_external_working_memory, True),
        ("PPO Importance Ratio", test_ppo_importance_ratio, True),
        ("PPO Training", test_ppo_training, True),  # Main PPO test
        ("Health Monitoring", test_health_monitoring, True),
        ("API Integration", test_api_integration, False)  # Optional - requires server
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, (test_name, test_func, required) in enumerate(test_cases, 1):
        print(f"\n🎬 [{i}/{len(test_cases)}] Starting: {test_name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            await test_func()
            duration = time.time() - start_time
            
            results.append({
                'name': test_name,
                'status': 'PASSED',
                'duration': duration,
                'required': required
            })
            
            print(f"✅ [{i}/{len(test_cases)}] PASSED: {test_name} ({duration:.1f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            status = 'FAILED' if required else 'SKIPPED'
            
            results.append({
                'name': test_name,
                'status': status, 
                'duration': duration,
                'error': str(e),
                'required': required
            })
            
            if required:
                print(f"❌ [{i}/{len(test_cases)}] FAILED: {test_name} - {e}")
            else:
                print(f"⏭️  [{i}/{len(test_cases)}] SKIPPED: {test_name} - {e}")
        
        # Brief pause between tests
        if i < len(test_cases):
            print("\n" + "=" * 60)
            await asyncio.sleep(1)
    
    # Final report
    total_duration = time.time() - total_start_time
    
    print(f"\n📊 Test Suite Summary")
    print("=" * 60)
    
    passed_tests = [r for r in results if r['status'] == 'PASSED']
    failed_tests = [r for r in results if r['status'] == 'FAILED']
    skipped_tests = [r for r in results if r['status'] == 'SKIPPED']
    
    print(f"📈 Overall Results:")
    print(f"  ✅ Passed:  {len(passed_tests)}/{len([r for r in results if r['required']])}")
    print(f"  ❌ Failed:  {len(failed_tests)}")
    print(f"  ⏭️  Skipped: {len(skipped_tests)}")
    print(f"  ⏱️  Total time: {total_duration:.1f}s")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        status_emoji = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⏭️"}[result['status']]
        required_text = "" if result['required'] else " (optional)"
        
        print(f"  {status_emoji} {result['name']}{required_text}")
        print(f"     Duration: {result['duration']:.1f}s")
        
        if result['status'] == 'FAILED':
            print(f"     Error: {result.get('error', 'Unknown error')}")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    
    ppo_test = next((r for r in results if 'PPO Training' in r['name']), None)
    if ppo_test and ppo_test['status'] == 'PASSED':
        print(f"  🎯 PPO Training: SUCCESSFUL - Check debug_reward_process.log for details")
    
    memory_tests = [r for r in results if 'Memory' in r['name'] and r['status'] == 'PASSED']
    print(f"  🧠 Memory Operations: {len(memory_tests)} tests passed")
    
    if any(r['status'] == 'FAILED' for r in results if r['required']):
        print(f"  ⚠️  Some required tests failed - check implementation")
        return False
    else:
        print(f"  ✅ All required tests passed - system functioning correctly")
        return True


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)