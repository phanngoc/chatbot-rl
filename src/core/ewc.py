"""
Elastic Weight Consolidation (EWC) Implementation
Tránh catastrophic forgetting bằng cách gắn penalty khi thay đổi trọng số quan trọng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict
import copy
import pickle
import os
from datetime import datetime


class FisherInformationCalculator:
    """Tính Fisher Information Matrix để xác định importance của weights"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def compute_fisher_information(self, 
                                 dataloader, 
                                 num_samples: Optional[int] = None,
                                 empirical: bool = True) -> Dict[str, torch.Tensor]:
        """
        Tính Fisher Information Matrix
        
        Args:
            dataloader: DataLoader chứa data để tính Fisher
            num_samples: Số samples để tính (None = tất cả)
            empirical: Dùng empirical Fisher (True) hay true Fisher (False)
        """
        self.model.eval()
        fisher_dict = {}
        
        # Initialize Fisher information dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if num_samples and sample_count >= num_samples:
                break
            
            # Forward pass
            inputs, targets = batch
            outputs = self.model(inputs)
            
            if empirical:
                # Empirical Fisher: sử dụng actual targets
                loss = F.cross_entropy(outputs, targets, reduction='sum')
            else:
                # True Fisher: sample từ model distribution
                sampled_targets = torch.multinomial(F.softmax(outputs, dim=1), 1).squeeze()
                loss = F.cross_entropy(outputs, sampled_targets, reduction='sum')
            
            # Compute gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients (Fisher Information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            sample_count += inputs.size(0)
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
        
        self.fisher_information = fisher_dict
        return fisher_dict
    
    def save_optimal_parameters(self) -> Dict[str, torch.Tensor]:
        """Lưu optimal parameters của task hiện tại"""
        optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal_params[name] = param.data.clone()
        
        self.optimal_params = optimal_params
        return optimal_params
    
    def get_importance_weights(self, 
                             threshold: float = 1e-6) -> Dict[str, torch.Tensor]:
        """Lấy importance weights từ Fisher Information"""
        importance_weights = {}
        
        for name, fisher in self.fisher_information.items():
            # Clamp Fisher values để tránh quá nhỏ
            clamped_fisher = torch.clamp(fisher, min=threshold)
            importance_weights[name] = clamped_fisher
        
        return importance_weights


class EWCLoss:
    """EWC Loss function"""
    
    def __init__(self, 
                 fisher_information: Dict[str, torch.Tensor],
                 optimal_params: Dict[str, torch.Tensor],
                 ewc_lambda: float = 1000.0):
        """
        Args:
            fisher_information: Fisher Information cho mỗi parameter
            optimal_params: Optimal parameters từ task trước
            ewc_lambda: Regularization strength
        """
        self.fisher_information = fisher_information
        self.optimal_params = optimal_params
        self.ewc_lambda = ewc_lambda
    
    def compute_ewc_penalty(self, model: nn.Module) -> torch.Tensor:
        """Tính EWC penalty term"""
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_information and name in self.optimal_params:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                
                # EWC penalty: Fisher * (param - optimal)^2
                penalty += (fisher * (param - optimal) ** 2).sum()
        
        return self.ewc_lambda * penalty
    
    def __call__(self, 
                 model: nn.Module, 
                 task_loss: torch.Tensor) -> torch.Tensor:
        """Tính total loss = task loss + EWC penalty"""
        ewc_penalty = self.compute_ewc_penalty(model)
        return task_loss + ewc_penalty


class MultiTaskEWC:
    """EWC cho multiple tasks"""
    
    def __init__(self, model: nn.Module, ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        
        # Storage cho multiple tasks
        self.task_fisher_info: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_optimal_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.task_importance: Dict[str, float] = {}
        
        self.current_task_id: Optional[str] = None
        self.fisher_calculator = FisherInformationCalculator(model)
    
    def start_new_task(self, task_id: str, importance: float = 1.0) -> None:
        """Bắt đầu task mới"""
        self.current_task_id = task_id
        self.task_importance[task_id] = importance
        
        print(f"Bắt đầu task mới: {task_id} với importance: {importance}")
    
    def finish_current_task(self, 
                          dataloader, 
                          num_samples: Optional[int] = None) -> None:
        """Kết thúc task hiện tại và tính Fisher Information"""
        if self.current_task_id is None:
            raise ValueError("Không có task nào đang active")
        
        task_id = self.current_task_id
        
        # Tính Fisher Information cho task hiện tại
        print(f"Tính Fisher Information cho task: {task_id}")
        fisher_info = self.fisher_calculator.compute_fisher_information(
            dataloader, num_samples
        )
        
        # Lưu optimal parameters
        optimal_params = self.fisher_calculator.save_optimal_parameters()
        
        # Store cho task này
        self.task_fisher_info[task_id] = fisher_info
        self.task_optimal_params[task_id] = optimal_params
        
        print(f"Hoàn thành task: {task_id}")
        self.current_task_id = None
    
    def compute_consolidated_penalty(self) -> torch.Tensor:
        """Tính consolidated EWC penalty từ tất cả previous tasks"""
        total_penalty = torch.tensor(0.0)
        
        for task_id in self.task_fisher_info:
            if task_id == self.current_task_id:
                continue  # Skip current task
            
            fisher_info = self.task_fisher_info[task_id]
            optimal_params = self.task_optimal_params[task_id]
            task_importance = self.task_importance[task_id]
            
            # Tính penalty cho task này
            task_penalty = torch.tensor(0.0)
            for name, param in self.model.named_parameters():
                if name in fisher_info and name in optimal_params:
                    fisher = fisher_info[name]
                    optimal = optimal_params[name]
                    
                    task_penalty += (fisher * (param - optimal) ** 2).sum()
            
            # Weight by task importance
            total_penalty += task_importance * task_penalty
        
        return self.ewc_lambda * total_penalty
    
    def get_ewc_loss(self, task_loss: torch.Tensor) -> torch.Tensor:
        """Lấy total loss với EWC penalty"""
        ewc_penalty = self.compute_consolidated_penalty()
        return task_loss + ewc_penalty
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Thống kê về các tasks"""
        stats = {
            "total_tasks": len(self.task_fisher_info),
            "current_task": self.current_task_id,
            "task_importance": self.task_importance.copy(),
            "ewc_lambda": self.ewc_lambda
        }
        
        # Parameter statistics
        if self.task_fisher_info:
            total_params = 0
            avg_fisher_magnitude = 0.0
            
            for task_id, fisher_info in self.task_fisher_info.items():
                for name, fisher in fisher_info.items():
                    total_params += fisher.numel()
                    avg_fisher_magnitude += fisher.mean().item()
            
            stats["total_parameters"] = total_params
            stats["avg_fisher_magnitude"] = avg_fisher_magnitude / len(self.task_fisher_info)
        
        return stats
    
    def save_ewc_data(self, filepath: str) -> None:
        """Lưu EWC data"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert tensors to CPU và serialize
        data = {
            "task_fisher_info": {
                task_id: {name: tensor.cpu() for name, tensor in fisher_info.items()}
                for task_id, fisher_info in self.task_fisher_info.items()
            },
            "task_optimal_params": {
                task_id: {name: tensor.cpu() for name, tensor in params.items()}
                for task_id, params in self.task_optimal_params.items()
            },
            "task_importance": self.task_importance,
            "ewc_lambda": self.ewc_lambda,
            "current_task_id": self.current_task_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_ewc_data(self, filepath: str) -> bool:
        """Load EWC data"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Move tensors to appropriate device
            device = next(self.model.parameters()).device
            
            self.task_fisher_info = {
                task_id: {name: tensor.to(device) for name, tensor in fisher_info.items()}
                for task_id, fisher_info in data["task_fisher_info"].items()
            }
            
            self.task_optimal_params = {
                task_id: {name: tensor.to(device) for name, tensor in params.items()}
                for task_id, params in data["task_optimal_params"].items()
            }
            
            self.task_importance = data["task_importance"]
            self.ewc_lambda = data["ewc_lambda"]
            self.current_task_id = data["current_task_id"]
            
            return True
        except Exception as e:
            print(f"Lỗi khi load EWC data: {e}")
            return False


class AdaptiveEWC(MultiTaskEWC):
    """Adaptive EWC với dynamic lambda adjustment"""
    
    def __init__(self, 
                 model: nn.Module, 
                 initial_lambda: float = 1000.0,
                 lambda_decay: float = 0.95,
                 min_lambda: float = 100.0):
        super().__init__(model, initial_lambda)
        self.initial_lambda = initial_lambda
        self.lambda_decay = lambda_decay
        self.min_lambda = min_lambda
        
        # Tracking performance
        self.task_performance: Dict[str, List[float]] = defaultdict(list)
        self.forgetting_threshold = 0.05  # 5% performance drop threshold
    
    def update_lambda_adaptive(self, 
                             current_performance: Dict[str, float]) -> None:
        """Cập nhật lambda dựa trên performance"""
        # Check for catastrophic forgetting
        forgetting_detected = False
        
        for task_id, current_perf in current_performance.items():
            if task_id in self.task_performance:
                historical_perf = self.task_performance[task_id]
                if historical_perf:
                    best_perf = max(historical_perf)
                    performance_drop = best_perf - current_perf
                    
                    if performance_drop > self.forgetting_threshold:
                        forgetting_detected = True
                        print(f"Phát hiện forgetting ở task {task_id}: {performance_drop:.3f}")
        
        # Adjust lambda
        if forgetting_detected:
            # Increase lambda để bảo vệ better
            self.ewc_lambda = min(self.ewc_lambda * 1.2, self.initial_lambda * 2)
            print(f"Tăng EWC lambda lên: {self.ewc_lambda}")
        else:
            # Decrease lambda để allow more flexibility
            self.ewc_lambda = max(self.ewc_lambda * self.lambda_decay, self.min_lambda)
        
        # Update performance history
        for task_id, perf in current_performance.items():
            self.task_performance[task_id].append(perf)
    
    def get_task_specific_lambda(self, task_id: str) -> float:
        """Lấy lambda specific cho task"""
        if task_id not in self.task_importance:
            return self.ewc_lambda
        
        # Scale lambda by task importance
        task_lambda = self.ewc_lambda * self.task_importance[task_id]
        return task_lambda


class EWCTrainer:
    """Trainer tích hợp EWC"""
    
    def __init__(self, 
                 model: nn.Module,
                 ewc_system: MultiTaskEWC,
                 optimizer,
                 device: str = "cpu"):
        self.model = model
        self.ewc_system = ewc_system
        self.optimizer = optimizer
        self.device = device
        
        self.training_history = []
    
    def train_with_ewc(self, 
                      dataloader,
                      num_epochs: int = 5,
                      task_id: Optional[str] = None) -> Dict[str, Any]:
        """Training với EWC regularization"""
        self.model.train()
        
        epoch_losses = []
        epoch_task_losses = []
        epoch_ewc_penalties = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_task_loss = 0.0
            total_ewc_penalty = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, targets)
                
                # EWC penalty
                ewc_penalty = self.ewc_system.compute_consolidated_penalty()
                total_loss_batch = task_loss + ewc_penalty
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += total_loss_batch.item()
                total_task_loss += task_loss.item()
                total_ewc_penalty += ewc_penalty.item()
                num_batches += 1
            
            # Average losses for epoch
            avg_loss = total_loss / num_batches
            avg_task_loss = total_task_loss / num_batches
            avg_ewc_penalty = total_ewc_penalty / num_batches
            
            epoch_losses.append(avg_loss)
            epoch_task_losses.append(avg_task_loss)
            epoch_ewc_penalties.append(avg_ewc_penalty)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {avg_loss:.4f}")
            print(f"  Task Loss: {avg_task_loss:.4f}")
            print(f"  EWC Penalty: {avg_ewc_penalty:.4f}")
        
        training_results = {
            "task_id": task_id,
            "num_epochs": num_epochs,
            "final_loss": epoch_losses[-1],
            "final_task_loss": epoch_task_losses[-1],
            "final_ewc_penalty": epoch_ewc_penalties[-1],
            "loss_history": epoch_losses,
            "task_loss_history": epoch_task_losses,
            "ewc_penalty_history": epoch_ewc_penalties
        }
        
        self.training_history.append(training_results)
        return training_results
    
    def evaluate_model(self, dataloader) -> Dict[str, float]:
        """Đánh giá model performance"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "correct": correct,
            "total": total
        }
    
    def continual_learning_cycle(self, 
                               task_dataloaders: Dict[str, Any],
                               num_epochs_per_task: int = 5) -> Dict[str, Any]:
        """Chạy full continual learning cycle"""
        results = {}
        
        for task_id, (train_loader, eval_loader) in task_dataloaders.items():
            print(f"\n=== Bắt đầu Task: {task_id} ===")
            
            # Start new task
            self.ewc_system.start_new_task(task_id)
            
            # Train on new task
            training_results = self.train_with_ewc(
                train_loader, 
                num_epochs_per_task, 
                task_id
            )
            
            # Evaluate on current task
            eval_results = self.evaluate_model(eval_loader)
            
            # Finish task và compute Fisher Information
            self.ewc_system.finish_current_task(train_loader, num_samples=1000)
            
            results[task_id] = {
                "training": training_results,
                "evaluation": eval_results
            }
            
            print(f"Task {task_id} - Accuracy: {eval_results['accuracy']:.3f}")
        
        # Final evaluation trên tất cả tasks
        print("\n=== Đánh giá cuối cùng trên tất cả tasks ===")
        final_results = {}
        for task_id, (_, eval_loader) in task_dataloaders.items():
            eval_results = self.evaluate_model(eval_loader)
            final_results[task_id] = eval_results
            print(f"Task {task_id} - Final Accuracy: {eval_results['accuracy']:.3f}")
        
        return {
            "task_results": results,
            "final_evaluation": final_results,
            "ewc_statistics": self.ewc_system.get_task_statistics()
        }
