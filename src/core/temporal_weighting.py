"""
Temporal Decay & Importance Weighting System
Quản lý trọng số của experiences theo thời gian và importance
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import pickle
import os
from collections import defaultdict
import math


@dataclass
class WeightedExperience:
    """Experience với temporal và importance weights"""
    id: str
    content: str
    context: str
    timestamp: datetime
    base_reward: float
    
    # Weighting factors
    temporal_weight: float = 1.0
    importance_weight: float = 1.0
    access_weight: float = 1.0
    quality_weight: float = 1.0
    
    # Access tracking
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_intervals: List[float] = field(default_factory=list)
    
    # Quality metrics
    user_feedback_score: float = 0.0
    effectiveness_score: float = 0.0  # Based on subsequent interactions
    novelty_score: float = 1.0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    source: str = "user_interaction"
    confidence_score: float = 1.0
    
    def get_combined_weight(self) -> float:
        """Tính combined weight từ tất cả factors"""
        return (self.temporal_weight * 
                self.importance_weight * 
                self.access_weight * 
                self.quality_weight)
    
    def get_weighted_reward(self) -> float:
        """Lấy reward đã được weight"""
        return self.base_reward * self.get_combined_weight()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "base_reward": self.base_reward,
            "temporal_weight": self.temporal_weight,
            "importance_weight": self.importance_weight,
            "access_weight": self.access_weight,
            "quality_weight": self.quality_weight,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "access_intervals": self.access_intervals,
            "user_feedback_score": self.user_feedback_score,
            "effectiveness_score": self.effectiveness_score,
            "novelty_score": self.novelty_score,
            "tags": self.tags,
            "source": self.source,
            "confidence_score": self.confidence_score
        }


class TemporalDecayFunction:
    """Các hàm decay theo thời gian"""
    
    @staticmethod
    def exponential_decay(days_old: float, 
                         decay_rate: float = 0.1,
                         min_weight: float = 0.01) -> float:
        """Exponential decay: w(t) = exp(-decay_rate * t)"""
        weight = math.exp(-decay_rate * days_old)
        return max(weight, min_weight)
    
    @staticmethod
    def power_law_decay(days_old: float,
                       alpha: float = 1.0,
                       min_weight: float = 0.01) -> float:
        """Power law decay: w(t) = 1 / (1 + t)^alpha"""
        weight = 1.0 / math.pow(1 + days_old, alpha)
        return max(weight, min_weight)
    
    @staticmethod
    def linear_decay(days_old: float,
                    decay_rate: float = 0.01,
                    min_weight: float = 0.01) -> float:
        """Linear decay: w(t) = max(1 - decay_rate * t, min_weight)"""
        weight = 1.0 - decay_rate * days_old
        return max(weight, min_weight)
    
    @staticmethod
    def forgetting_curve(days_old: float,
                        strength: float = 1.0,
                        decay_rate: float = 0.5) -> float:
        """Ebbinghaus forgetting curve: R = e^(-t/S)"""
        weight = math.exp(-days_old / (strength / decay_rate))
        return max(weight, 0.01)
    
    @staticmethod
    def stepped_decay(days_old: float,
                     thresholds: List[Tuple[float, float]] = None) -> float:
        """Stepped decay với different rates cho different time periods"""
        if thresholds is None:
            thresholds = [(7, 1.0), (30, 0.7), (90, 0.4), (365, 0.1)]
        
        for threshold_days, weight in thresholds:
            if days_old <= threshold_days:
                return weight
        
        return thresholds[-1][1]  # Return last weight


class ImportanceCalculator:
    """Tính importance weight cho experiences"""
    
    def __init__(self):
        self.feedback_weight = 0.3
        self.reward_weight = 0.4
        self.novelty_weight = 0.2
        self.effectiveness_weight = 0.1
    
    def calculate_importance(self, experience: WeightedExperience) -> float:
        """Tính importance weight tổng hợp"""
        # Reward-based importance
        reward_importance = self._reward_to_importance(experience.base_reward)
        
        # Feedback-based importance
        feedback_importance = self._feedback_to_importance(experience.user_feedback_score)
        
        # Novelty-based importance
        novelty_importance = experience.novelty_score
        
        # Effectiveness-based importance
        effectiveness_importance = experience.effectiveness_score
        
        # Weighted combination
        total_importance = (
            self.reward_weight * reward_importance +
            self.feedback_weight * feedback_importance +
            self.novelty_weight * novelty_importance +
            self.effectiveness_weight * effectiveness_importance
        )
        
        return max(0.1, min(3.0, total_importance))  # Clamp between 0.1 and 3.0
    
    def _reward_to_importance(self, reward: float) -> float:
        """Convert reward to importance score"""
        # Normalize reward to [0, 1] và scale
        normalized_reward = (reward + 1) / 2  # Assume reward in [-1, 1]
        return 0.5 + normalized_reward  # Range [0.5, 1.5]
    
    def _feedback_to_importance(self, feedback_score: float) -> float:
        """Convert user feedback to importance score"""
        # feedback_score should be in [-1, 1]
        if feedback_score > 0.5:
            return 1.5  # High positive feedback
        elif feedback_score < -0.5:
            return 1.2  # High negative feedback (still important to remember)
        else:
            return 1.0  # Neutral feedback


class AccessPatternAnalyzer:
    """Phân tích patterns của memory access"""
    
    def __init__(self):
        self.access_boost_factor = 1.2
        self.recency_boost_factor = 1.1
        self.frequency_weight = 0.6
        self.recency_weight = 0.4
    
    def calculate_access_weight(self, experience: WeightedExperience) -> float:
        """Tính access weight dựa trên access patterns"""
        # Frequency component
        frequency_score = self._calculate_frequency_score(experience)
        
        # Recency component
        recency_score = self._calculate_recency_score(experience)
        
        # Combined access weight
        access_weight = (
            self.frequency_weight * frequency_score +
            self.recency_weight * recency_score
        )
        
        return max(0.5, min(2.0, access_weight))
    
    def _calculate_frequency_score(self, experience: WeightedExperience) -> float:
        """Tính score dựa trên frequency of access"""
        if experience.access_count == 0:
            return 1.0
        
        # Log-scale để avoid extreme values
        frequency_score = 1.0 + 0.1 * math.log(1 + experience.access_count)
        return min(frequency_score, 2.0)
    
    def _calculate_recency_score(self, experience: WeightedExperience) -> float:
        """Tính score dựa trên recency of access"""
        if experience.access_count == 0:
            return 1.0
        
        hours_since_access = (datetime.now() - experience.last_accessed).total_seconds() / 3600
        
        # Boost recently accessed memories
        if hours_since_access < 24:
            return self.recency_boost_factor
        elif hours_since_access < 168:  # 1 week
            return 1.0
        else:
            return 0.8
    
    def analyze_access_intervals(self, experience: WeightedExperience) -> Dict[str, float]:
        """Phân tích intervals between accesses"""
        if len(experience.access_intervals) < 2:
            return {"regularity": 1.0, "predictability": 1.0}
        
        intervals = experience.access_intervals
        
        # Regularity: coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        regularity = 1.0 / (1.0 + std_interval / (mean_interval + 1e-8))
        
        # Predictability: based on trend
        if len(intervals) >= 3:
            # Simple trend analysis
            recent_intervals = intervals[-3:]
            trend = np.polyfit(range(len(recent_intervals)), recent_intervals, 1)[0]
            predictability = 1.0 / (1.0 + abs(trend))
        else:
            predictability = 1.0
        
        return {
            "regularity": regularity,
            "predictability": predictability,
            "mean_interval": mean_interval,
            "std_interval": std_interval
        }


class QualityAssessment:
    """Đánh giá quality của experiences"""
    
    def __init__(self):
        self.coherence_weight = 0.3
        self.relevance_weight = 0.4
        self.uniqueness_weight = 0.3
    
    def calculate_quality_weight(self, 
                               experience: WeightedExperience,
                               context_experiences: List[WeightedExperience] = None) -> float:
        """Tính quality weight"""
        # Coherence score (simplified)
        coherence_score = self._assess_coherence(experience)
        
        # Relevance score
        relevance_score = self._assess_relevance(experience, context_experiences)
        
        # Uniqueness score
        uniqueness_score = self._assess_uniqueness(experience, context_experiences)
        
        quality_weight = (
            self.coherence_weight * coherence_score +
            self.relevance_weight * relevance_score +
            self.uniqueness_weight * uniqueness_score
        )
        
        return max(0.5, min(2.0, quality_weight))
    
    def _assess_coherence(self, experience: WeightedExperience) -> float:
        """Đánh giá coherence của experience"""
        # Simplified coherence based on content length và structure
        content_length = len(experience.content.split())
        context_length = len(experience.context.split())
        
        # Prefer experiences with reasonable length
        if 5 <= content_length <= 100 and context_length > 0:
            return 1.2
        elif content_length < 5 or content_length > 200:
            return 0.8
        else:
            return 1.0
    
    def _assess_relevance(self, 
                         experience: WeightedExperience,
                         context_experiences: List[WeightedExperience] = None) -> float:
        """Đánh giá relevance với context"""
        if not context_experiences:
            return 1.0
        
        # Simple relevance based on common words (trong thực tế sẽ dùng embeddings)
        experience_words = set(experience.content.lower().split())
        
        relevance_scores = []
        for ctx_exp in context_experiences[-5:]:  # Check recent 5 experiences
            ctx_words = set(ctx_exp.content.lower().split())
            if experience_words and ctx_words:
                overlap = len(experience_words.intersection(ctx_words))
                union = len(experience_words.union(ctx_words))
                jaccard_similarity = overlap / union if union > 0 else 0
                relevance_scores.append(jaccard_similarity)
        
        if relevance_scores:
            avg_relevance = np.mean(relevance_scores)
            return 0.8 + 0.4 * avg_relevance  # Scale to [0.8, 1.2]
        
        return 1.0
    
    def _assess_uniqueness(self, 
                          experience: WeightedExperience,
                          context_experiences: List[WeightedExperience] = None) -> float:
        """Đánh giá uniqueness của experience"""
        if not context_experiences:
            return 1.0
        
        # Check for similar experiences
        experience_words = set(experience.content.lower().split())
        
        max_similarity = 0.0
        for ctx_exp in context_experiences:
            if ctx_exp.id == experience.id:
                continue
            
            ctx_words = set(ctx_exp.content.lower().split())
            if experience_words and ctx_words:
                overlap = len(experience_words.intersection(ctx_words))
                union = len(experience_words.union(ctx_words))
                similarity = overlap / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)
        
        # Higher uniqueness = lower similarity
        uniqueness = 1.0 - max_similarity
        return 0.7 + 0.6 * uniqueness  # Scale to [0.7, 1.3]


class TemporalWeightingSystem:
    """Main system cho temporal decay và importance weighting"""
    
    def __init__(self,
                 decay_function: str = "exponential",
                 decay_params: Dict[str, float] = None,
                 update_interval_hours: int = 6):
        
        self.decay_function_name = decay_function
        self.decay_params = decay_params or {"decay_rate": 0.05, "min_weight": 0.01}
        self.update_interval = timedelta(hours=update_interval_hours)
        self.last_update = datetime.now()
        
        # Initialize components
        self.importance_calculator = ImportanceCalculator()
        self.access_analyzer = AccessPatternAnalyzer()
        self.quality_assessor = QualityAssessment()
        
        # Experience storage
        self.experiences: Dict[str, WeightedExperience] = {}
        self.weight_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Statistics
        self.update_stats = {
            "total_updates": 0,
            "avg_weight_change": 0.0,
            "experiences_pruned": 0,
            "last_update_time": self.last_update
        }
    
    def add_experience(self, 
                      experience_id: str,
                      content: str,
                      context: str = "",
                      reward: float = 0.0,
                      tags: List[str] = None,
                      source: str = "user_interaction") -> WeightedExperience:
        """Thêm experience mới với initial weights"""
        
        experience = WeightedExperience(
            id=experience_id,
            content=content,
            context=context,
            timestamp=datetime.now(),
            base_reward=reward,
            tags=tags or [],
            source=source
        )
        
        # Calculate initial weights
        self._update_experience_weights(experience)
        
        self.experiences[experience_id] = experience
        
        # Record initial weight
        self.weight_history[experience_id].append(
            (experience.timestamp, experience.get_combined_weight())
        )
        
        return experience
    
    def update_experience_feedback(self,
                                 experience_id: str,
                                 user_feedback: float,
                                 effectiveness_score: float = None) -> bool:
        """Cập nhật feedback cho experience"""
        if experience_id not in self.experiences:
            return False
        
        experience = self.experiences[experience_id]
        experience.user_feedback_score = user_feedback
        
        if effectiveness_score is not None:
            experience.effectiveness_score = effectiveness_score
        
        # Recalculate weights
        self._update_experience_weights(experience)
        
        return True
    
    def access_experience(self, experience_id: str) -> Optional[WeightedExperience]:
        """Access experience và update access patterns"""
        if experience_id not in self.experiences:
            return None
        
        experience = self.experiences[experience_id]
        
        # Update access information
        current_time = datetime.now()
        if experience.access_count > 0:
            # Calculate interval since last access
            interval = (current_time - experience.last_accessed).total_seconds() / 3600  # hours
            experience.access_intervals.append(interval)
            
            # Keep only recent intervals
            if len(experience.access_intervals) > 20:
                experience.access_intervals = experience.access_intervals[-15:]
        
        experience.access_count += 1
        experience.last_accessed = current_time
        
        # Update weights
        self._update_experience_weights(experience)
        
        return experience
    
    def _update_experience_weights(self, experience: WeightedExperience) -> None:
        """Update tất cả weights cho experience"""
        current_time = datetime.now()
        
        # Temporal weight
        days_old = (current_time - experience.timestamp).total_seconds() / (24 * 3600)
        experience.temporal_weight = self._calculate_temporal_weight(days_old)
        
        # Importance weight
        experience.importance_weight = self.importance_calculator.calculate_importance(experience)
        
        # Access weight
        experience.access_weight = self.access_analyzer.calculate_access_weight(experience)
        
        # Quality weight
        context_experiences = list(self.experiences.values())
        experience.quality_weight = self.quality_assessor.calculate_quality_weight(
            experience, context_experiences
        )
    
    def _calculate_temporal_weight(self, days_old: float) -> float:
        """Tính temporal weight dựa trên decay function"""
        decay_func = getattr(TemporalDecayFunction, f"{self.decay_function_name}_decay")
        return decay_func(days_old, **self.decay_params)
    
    def batch_update_weights(self) -> Dict[str, Any]:
        """Update weights cho tất cả experiences"""
        if datetime.now() - self.last_update < self.update_interval:
            return {"status": "skipped", "reason": "too_soon"}
        
        updated_count = 0
        total_weight_change = 0.0
        
        for experience in self.experiences.values():
            old_weight = experience.get_combined_weight()
            self._update_experience_weights(experience)
            new_weight = experience.get_combined_weight()
            
            weight_change = abs(new_weight - old_weight)
            total_weight_change += weight_change
            updated_count += 1
            
            # Record weight change
            self.weight_history[experience.id].append((datetime.now(), new_weight))
            
            # Keep history manageable
            if len(self.weight_history[experience.id]) > 100:
                self.weight_history[experience.id] = self.weight_history[experience.id][-50:]
        
        # Update statistics
        self.update_stats["total_updates"] += 1
        self.update_stats["avg_weight_change"] = total_weight_change / max(updated_count, 1)
        self.update_stats["last_update_time"] = datetime.now()
        self.last_update = datetime.now()
        
        return {
            "status": "completed",
            "experiences_updated": updated_count,
            "avg_weight_change": self.update_stats["avg_weight_change"],
            "total_weight_change": total_weight_change
        }
    
    def get_weighted_experiences(self, 
                               top_k: int = None,
                               min_weight: float = 0.1,
                               tags_filter: List[str] = None) -> List[WeightedExperience]:
        """Lấy experiences được sort theo weight"""
        filtered_experiences = []
        
        for experience in self.experiences.values():
            # Apply filters
            if experience.get_combined_weight() < min_weight:
                continue
            
            if tags_filter and not any(tag in experience.tags for tag in tags_filter):
                continue
            
            filtered_experiences.append(experience)
        
        # Sort by combined weight
        sorted_experiences = sorted(
            filtered_experiences,
            key=lambda x: x.get_combined_weight(),
            reverse=True
        )
        
        if top_k:
            return sorted_experiences[:top_k]
        
        return sorted_experiences
    
    def prune_low_weight_experiences(self, 
                                   weight_threshold: float = 0.05,
                                   age_threshold_days: int = 365) -> int:
        """Xóa experiences với weight thấp hoặc quá cũ"""
        current_time = datetime.now()
        experiences_to_remove = []
        
        for exp_id, experience in self.experiences.items():
            days_old = (current_time - experience.timestamp).total_seconds() / (24 * 3600)
            combined_weight = experience.get_combined_weight()
            
            should_remove = (
                combined_weight < weight_threshold or
                (days_old > age_threshold_days and combined_weight < 0.2)
            )
            
            if should_remove:
                experiences_to_remove.append(exp_id)
        
        # Remove experiences
        for exp_id in experiences_to_remove:
            del self.experiences[exp_id]
            if exp_id in self.weight_history:
                del self.weight_history[exp_id]
        
        pruned_count = len(experiences_to_remove)
        self.update_stats["experiences_pruned"] += pruned_count
        
        return pruned_count
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Thống kê về weights"""
        if not self.experiences:
            return {"total_experiences": 0}
        
        weights = [exp.get_combined_weight() for exp in self.experiences.values()]
        temporal_weights = [exp.temporal_weight for exp in self.experiences.values()]
        importance_weights = [exp.importance_weight for exp in self.experiences.values()]
        access_weights = [exp.access_weight for exp in self.experiences.values()]
        quality_weights = [exp.quality_weight for exp in self.experiences.values()]
        
        return {
            "total_experiences": len(self.experiences),
            "weight_distribution": {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights),
                "median": np.median(weights)
            },
            "component_weights": {
                "temporal": {"mean": np.mean(temporal_weights), "std": np.std(temporal_weights)},
                "importance": {"mean": np.mean(importance_weights), "std": np.std(importance_weights)},
                "access": {"mean": np.mean(access_weights), "std": np.std(access_weights)},
                "quality": {"mean": np.mean(quality_weights), "std": np.std(quality_weights)}
            },
            "high_weight_experiences": sum(1 for w in weights if w > 1.5),
            "low_weight_experiences": sum(1 for w in weights if w < 0.5),
            "update_statistics": self.update_stats
        }
    
    def save_system(self, filepath: str) -> None:
        """Lưu system state"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            "decay_function_name": self.decay_function_name,
            "decay_params": self.decay_params,
            "update_interval_hours": self.update_interval.total_seconds() / 3600,
            "experiences": {
                exp_id: exp.to_dict() for exp_id, exp in self.experiences.items()
            },
            "weight_history": {
                exp_id: [(dt.isoformat(), weight) for dt, weight in history]
                for exp_id, history in self.weight_history.items()
            },
            "update_stats": {
                **self.update_stats,
                "last_update_time": self.update_stats["last_update_time"].isoformat()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_system(self, filepath: str) -> bool:
        """Load system state"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load configuration
            self.decay_function_name = data["decay_function_name"]
            self.decay_params = data["decay_params"]
            self.update_interval = timedelta(hours=data["update_interval_hours"])
            
            # Load experiences
            self.experiences = {}
            for exp_id, exp_data in data["experiences"].items():
                exp_data["timestamp"] = datetime.fromisoformat(exp_data["timestamp"])
                exp_data["last_accessed"] = datetime.fromisoformat(exp_data["last_accessed"])
                
                experience = WeightedExperience(**exp_data)
                self.experiences[exp_id] = experience
            
            # Load weight history
            self.weight_history = defaultdict(list)
            for exp_id, history in data["weight_history"].items():
                self.weight_history[exp_id] = [
                    (datetime.fromisoformat(dt), weight) for dt, weight in history
                ]
            
            # Load update stats
            self.update_stats = data["update_stats"]
            self.update_stats["last_update_time"] = datetime.fromisoformat(
                self.update_stats["last_update_time"]
            )
            
            return True
        except Exception as e:
            print(f"Lỗi khi load system: {e}")
            return False
