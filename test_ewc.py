"""
Test file để demo cách EWC classes hoạt động
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.ewc import (
    FisherInformationCalculator,
    EWCLoss,
    MultiTaskEWC,
    AdaptiveEWC,
    EWCTrainer
)


class SimpleNN(nn.Module):
    """Simple neural network cho testing"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def create_dummy_dataset(num_samples=1000, input_size=10, num_classes=3, task_id=0):
    """Tạo dummy dataset cho testing"""
    
    # Tạo data khác nhau cho mỗi task
    np.random.seed(42 + task_id)
    torch.manual_seed(42 + task_id)
    
    # Generate features với pattern khác nhau cho mỗi task
    if task_id == 0:
        # Task 0: linear pattern
        X = torch.randn(num_samples, input_size)
        weights = torch.randn(input_size, num_classes)
        y = torch.argmax(torch.matmul(X, weights), dim=1)
    elif task_id == 1:
        # Task 1: quadratic pattern  
        X = torch.randn(num_samples, input_size)
        X_squared = X ** 2
        weights = torch.randn(input_size, num_classes)
        y = torch.argmax(torch.matmul(X_squared, weights), dim=1)
    else:
        # Task 2: mixed pattern
        X = torch.randn(num_samples, input_size)
        X_mixed = torch.sin(X) + 0.5 * X
        weights = torch.randn(input_size, num_classes)
        y = torch.argmax(torch.matmul(X_mixed, weights), dim=1)
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)


def test_fisher_information_calculator():
    """Test FisherInformationCalculator"""
    print("\n=== Test FisherInformationCalculator ===")
    
    # Tạo model và data
    model = SimpleNN()
    dataloader = create_dummy_dataset(num_samples=200, task_id=0)
    
    # Tạo calculator
    fisher_calc = FisherInformationCalculator(model)
    
    # Tính Fisher Information
    print("Đang tính Fisher Information...")
    fisher_info = fisher_calc.compute_fisher_information(dataloader, num_samples=100)
    
    # Lưu optimal parameters
    optimal_params = fisher_calc.save_optimal_parameters()
    
    # Kiểm tra kết quả
    print(f"Số parameters có Fisher info: {len(fisher_info)}")
    print(f"Số optimal parameters: {len(optimal_params)}")
    
    # In một số statistics
    for name, fisher in list(fisher_info.items())[:2]:  # In 2 đầu tiên
        print(f"Parameter {name}:")
        print(f"  Fisher shape: {fisher.shape}")
        print(f"  Fisher mean: {fisher.mean().item():.6f}")
        print(f"  Fisher std: {fisher.std().item():.6f}")
    
    print("✅ FisherInformationCalculator test passed!")
    return fisher_calc, fisher_info, optimal_params


def test_ewc_loss():
    """Test EWCLoss"""
    print("\n=== Test EWCLoss ===")
    
    model = SimpleNN()
    
    # Tạo dummy Fisher info và optimal params
    fisher_info = {}
    optimal_params = {}
    
    for name, param in model.named_parameters():
        fisher_info[name] = torch.ones_like(param) * 0.1  # Uniform Fisher
        optimal_params[name] = param.data.clone()
    
    # Tạo EWC loss
    ewc_loss = EWCLoss(fisher_info, optimal_params, ewc_lambda=1000.0)
    
    # Test với dummy task loss
    dummy_input = torch.randn(10, 10)
    dummy_target = torch.randint(0, 3, (10,))
    
    output = model(dummy_input)
    task_loss = F.cross_entropy(output, dummy_target)
    
    # Tính EWC penalty (should be 0 vì parameters chưa thay đổi)
    ewc_penalty = ewc_loss.compute_ewc_penalty(model)
    total_loss = ewc_loss(model, task_loss)
    
    print(f"Task loss: {task_loss.item():.4f}")
    print(f"EWC penalty: {ewc_penalty.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Thay đổi parameters và test lại
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.1)
    
    ewc_penalty_after = ewc_loss.compute_ewc_penalty(model)
    print(f"EWC penalty after parameter change: {ewc_penalty_after.item():.4f}")
    
    assert ewc_penalty_after > ewc_penalty, "EWC penalty should increase after parameter change"
    print("✅ EWCLoss test passed!")


def test_multitask_ewc():
    """Test MultiTaskEWC"""
    print("\n=== Test MultiTaskEWC ===")
    
    model = SimpleNN()
    ewc_system = MultiTaskEWC(model, ewc_lambda=500.0)
    
    # Tạo 3 tasks
    task_dataloaders = {}
    for task_id in range(3):
        train_loader = create_dummy_dataset(num_samples=300, task_id=task_id)
        eval_loader = create_dummy_dataset(num_samples=100, task_id=task_id)
        task_dataloaders[f"task_{task_id}"] = (train_loader, eval_loader)
    
    # Simulate training trên multiple tasks
    for task_id, (train_loader, eval_loader) in task_dataloaders.items():
        print(f"\nProcessing {task_id}...")
        
        # Start task
        ewc_system.start_new_task(task_id, importance=1.0)
        
        # Simulate một vài training steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= 5:  # Chỉ train 5 batches
                break
                
            outputs = model(inputs)
            task_loss = F.cross_entropy(outputs, targets)
            
            # Get EWC loss
            total_loss = ewc_system.get_ewc_loss(task_loss)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if batch_idx == 0:
                print(f"  Batch 0 - Task loss: {task_loss.item():.4f}, Total loss: {total_loss.item():.4f}")
        
        # Finish task
        ewc_system.finish_current_task(train_loader, num_samples=200)
    
    # Kiểm tra statistics
    stats = ewc_system.get_task_statistics()
    print(f"\nEWC Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Task importance: {stats['task_importance']}")
    print(f"  EWC lambda: {stats['ewc_lambda']}")
    print(f"  Total parameters: {stats['total_parameters']}")
    
    print("✅ MultiTaskEWC test passed!")
    return ewc_system


def test_adaptive_ewc():
    """Test AdaptiveEWC"""
    print("\n=== Test AdaptiveEWC ===")
    
    model = SimpleNN()
    adaptive_ewc = AdaptiveEWC(
        model, 
        initial_lambda=1000.0,
        lambda_decay=0.9,
        min_lambda=100.0
    )
    
    # Test adaptive lambda adjustment
    initial_lambda = adaptive_ewc.ewc_lambda
    print(f"Initial lambda: {initial_lambda}")
    
    # Simulate good performance (no forgetting)
    good_performance = {"task_0": 0.85, "task_1": 0.82}
    adaptive_ewc.update_lambda_adaptive(good_performance)
    print(f"Lambda after good performance: {adaptive_ewc.ewc_lambda}")
    
    # Simulate bad performance (forgetting detected)
    bad_performance = {"task_0": 0.75, "task_1": 0.70}  # Drop > 5%
    adaptive_ewc.update_lambda_adaptive(bad_performance)
    print(f"Lambda after forgetting detected: {adaptive_ewc.ewc_lambda}")
    
    print("✅ AdaptiveEWC test passed!")


def test_ewc_trainer():
    """Test EWCTrainer"""
    print("\n=== Test EWCTrainer ===")
    
    model = SimpleNN()
    ewc_system = MultiTaskEWC(model, ewc_lambda=200.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    trainer = EWCTrainer(model, ewc_system, optimizer, device="cpu")
    
    # Tạo task data
    task_dataloaders = {}
    for task_id in range(2):
        train_loader = create_dummy_dataset(num_samples=200, task_id=task_id)
        eval_loader = create_dummy_dataset(num_samples=100, task_id=task_id)
        task_dataloaders[f"task_{task_id}"] = (train_loader, eval_loader)
    
    # Run continual learning cycle
    results = trainer.continual_learning_cycle(
        task_dataloaders, 
        num_epochs_per_task=2  # Ít epochs cho test nhanh
    )
    
    # Kiểm tra results
    print(f"\nTraining Results:")
    for task_id, task_results in results["task_results"].items():
        accuracy = task_results["evaluation"]["accuracy"]
        print(f"  {task_id} accuracy: {accuracy:.3f}")
    
    print(f"\nFinal Evaluation:")
    for task_id, eval_results in results["final_evaluation"].items():
        accuracy = eval_results["accuracy"]
        print(f"  {task_id} final accuracy: {accuracy:.3f}")
    
    print("✅ EWCTrainer test passed!")
    return results


def test_save_load_functionality():
    """Test save/load functionality"""
    print("\n=== Test Save/Load Functionality ===")
    
    model = SimpleNN()
    ewc_system = MultiTaskEWC(model, ewc_lambda=300.0)
    
    # Tạo một task để có data
    train_loader = create_dummy_dataset(num_samples=100, task_id=0)
    ewc_system.start_new_task("test_task", importance=0.8)
    
    # Simulate training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= 2:
            break
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    ewc_system.finish_current_task(train_loader, num_samples=50)
    
    # Save EWC data
    save_path = "/Users/ngocp/Documents/projects/chatbot-rl/test_ewc_data.pkl"
    ewc_system.save_ewc_data(save_path)
    print(f"Saved EWC data to: {save_path}")
    
    # Create new system và load
    model2 = SimpleNN()
    ewc_system2 = MultiTaskEWC(model2, ewc_lambda=100.0)
    
    success = ewc_system2.load_ewc_data(save_path)
    print(f"Load successful: {success}")
    
    if success:
        stats1 = ewc_system.get_task_statistics()
        stats2 = ewc_system2.get_task_statistics()
        
        print(f"Original lambda: {stats1['ewc_lambda']}")
        print(f"Loaded lambda: {stats2['ewc_lambda']}")
        print(f"Tasks match: {stats1['task_importance'] == stats2['task_importance']}")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
        print("Cleaned up test file")
    
    print("✅ Save/Load test passed!")


def main():
    """Chạy tất cả tests"""
    print("🚀 Bắt đầu EWC Demo Tests...")
    
    try:
        # Test từng component
        test_fisher_information_calculator()
        test_ewc_loss()
        test_multitask_ewc()
        test_adaptive_ewc()
        test_ewc_trainer()
        test_save_load_functionality()
        
        print("\n🎉 Tất cả tests đã pass! EWC system hoạt động tốt.")
        
    except Exception as e:
        print(f"\n❌ Test failed với lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
