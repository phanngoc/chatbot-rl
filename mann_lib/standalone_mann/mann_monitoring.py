"""
MANN Monitoring và Pager System
Giám sát hiệu suất và cảnh báo cho production environment
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque, defaultdict
import aiohttp
import psutil
import threading
from enum import Enum


class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    timestamp: datetime
    memory_utilization: float
    processing_time: float
    query_count: int
    error_count: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float


class PagerSystem:
    """Pager system for production alerts"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.alert_history: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, Alert] = {}
        self.logger = logging.getLogger("PagerSystem")
        
        # Alert thresholds
        self.thresholds = {
            "memory_utilization": 0.9,
            "processing_time": 5.0,  # seconds
            "error_rate": 0.1,  # 10%
            "cpu_usage": 0.8,
            "memory_usage": 0.8,
            "disk_usage": 0.9
        }
        
        # Rate limiting
        self.rate_limits = {
            AlertLevel.INFO: 60,  # 1 minute
            AlertLevel.WARNING: 30,  # 30 seconds
            AlertLevel.ERROR: 10,  # 10 seconds
            AlertLevel.CRITICAL: 0  # No rate limit
        }
        self.last_alert_times: Dict[str, datetime] = {}
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        try:
            # Check rate limiting
            if self._is_rate_limited(alert):
                return False
            
            # Add to history
            self.alert_history.append(alert)
            self.active_alerts[alert.id] = alert
            
            # Send via webhook if configured
            if self.webhook_url:
                try:
                    await self._send_webhook(alert)
                except Exception as e:
                    self.logger.error(f"Failed to send webhook: {e}")
            
            # Log alert
            self.logger.log(
                self._get_log_level(alert.level),
                f"ALERT [{alert.level.value.upper()}] {alert.message} - {alert.source}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
            return False
    
    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert is rate limited"""
        rate_limit = self.rate_limits.get(alert.level, 60)
        if rate_limit == 0:
            return False
        
        alert_key = f"{alert.level.value}:{alert.source}"
        last_time = self.last_alert_times.get(alert_key)
        
        if last_time:
            time_diff = (datetime.now() - last_time).total_seconds()
            if time_diff < rate_limit:
                return True
        
        self.last_alert_times[alert_key] = datetime.now()
        return False
    
    def _get_log_level(self, alert_level: AlertLevel) -> int:
        """Get logging level for alert"""
        mapping = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }
        return mapping.get(alert_level, logging.INFO)
    
    async def _send_webhook(self, alert: Alert) -> None:
        """Send alert via webhook"""
        payload = {
            "alert_id": alert.id,
            "level": alert.level.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "source": alert.source,
            "metadata": alert.metadata
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Webhook failed with status {response.status}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]


class MANNMonitor:
    """MANN system monitor"""
    
    def __init__(self, config):
        self.config = config
        self.pager = PagerSystem(config.pager_webhook_url) if config.enable_pager else None
        self.logger = logging.getLogger("MANNMonitor")
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.query_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.query_count = 0
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Performance tracking
        self.performance_stats = {
            "avg_processing_time": 0.0,
            "max_processing_time": 0.0,
            "error_rate": 0.0,
            "memory_utilization": 0.0,
            "query_throughput": 0.0
        }
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("MANN monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("MANN monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check for alerts
                self._check_alerts(metrics)
                
                # Update performance stats
                self._update_performance_stats()
                
                # Sleep for interval
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect system performance metrics"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            memory_utilization=self.performance_stats.get("memory_utilization", 0.0),
            processing_time=self.performance_stats.get("avg_processing_time", 0.0),
            query_count=self.query_count,
            error_count=self.error_count,
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent
        )
    
    def _send_alert_safe(self, alert: Alert):
        """Send alert safely from any thread"""
        if not self.pager:
            return
        
        try:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self.pager.send_alert(alert), loop)
        except RuntimeError:
            # No event loop running, skip alert
            pass
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for alert conditions"""
        if not self.pager:
            return
        
        # Memory utilization alert
        if metrics.memory_utilization > self.pager.thresholds["memory_utilization"]:
            alert = Alert(
                id=f"memory_util_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High memory utilization: {metrics.memory_utilization:.2%}",
                timestamp=datetime.now(),
                source="memory_monitor",
                metadata={"utilization": metrics.memory_utilization}
            )
            self._send_alert_safe(alert)
        
        # Processing time alert
        if metrics.processing_time > self.pager.thresholds["processing_time"]:
            alert = Alert(
                id=f"processing_time_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High processing time: {metrics.processing_time:.2f}s",
                timestamp=datetime.now(),
                source="performance_monitor",
                metadata={"processing_time": metrics.processing_time}
            )
            self._send_alert_safe(alert)
        
        # Error rate alert
        if self.query_count > 0:
            error_rate = self.error_count / self.query_count
            if error_rate > self.pager.thresholds["error_rate"]:
                alert = Alert(
                    id=f"error_rate_{int(time.time())}",
                    level=AlertLevel.ERROR,
                    message=f"High error rate: {error_rate:.2%}",
                    timestamp=datetime.now(),
                    source="error_monitor",
                    metadata={"error_rate": error_rate, "error_count": self.error_count}
                )
                self._send_alert_safe(alert)
        
        # System resource alerts
        if metrics.cpu_usage > self.pager.thresholds["cpu_usage"] * 100:
            alert = Alert(
                id=f"cpu_usage_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                timestamp=datetime.now(),
                source="system_monitor",
                metadata={"cpu_usage": metrics.cpu_usage}
            )
            self._send_alert_safe(alert)
        
        if metrics.memory_usage > self.pager.thresholds["memory_usage"] * 100:
            alert = Alert(
                id=f"memory_usage_{int(time.time())}",
                level=AlertLevel.WARNING,
                message=f"High system memory usage: {metrics.memory_usage:.1f}%",
                timestamp=datetime.now(),
                source="system_monitor",
                metadata={"memory_usage": metrics.memory_usage}
            )
            self._send_alert_safe(alert)
        
        if metrics.disk_usage > self.pager.thresholds["disk_usage"] * 100:
            alert = Alert(
                id=f"disk_usage_{int(time.time())}",
                level=AlertLevel.CRITICAL,
                message=f"High disk usage: {metrics.disk_usage:.1f}%",
                timestamp=datetime.now(),
                source="system_monitor",
                metadata={"disk_usage": metrics.disk_usage}
            )
            self._send_alert_safe(alert)
    
    def _update_performance_stats(self):
        """Update performance statistics"""
        if not self.query_times:
            return
        
        # Calculate statistics
        times = list(self.query_times)
        self.performance_stats["avg_processing_time"] = sum(times) / len(times)
        self.performance_stats["max_processing_time"] = max(times)
        
        if self.query_count > 0:
            self.performance_stats["error_rate"] = self.error_count / self.query_count
        
        # Calculate throughput (queries per minute)
        if len(self.metrics_history) >= 2:
            recent_metrics = list(self.metrics_history)[-2:]
            time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
            query_diff = recent_metrics[-1].query_count - recent_metrics[0].query_count
            if time_diff > 0:
                self.performance_stats["query_throughput"] = (query_diff / time_diff) * 60
    
    def record_query(self, processing_time: float, memory_count: int):
        """Record query metrics"""
        self.query_count += 1
        self.query_times.append(processing_time)
        self.performance_stats["memory_utilization"] = memory_count / self.config.memory_size
    
    def record_error(self):
        """Record error"""
        self.error_count += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.performance_stats,
            "total_queries": self.query_count,
            "total_errors": self.error_count,
            "uptime": time.time() - (self.metrics_history[0].timestamp.timestamp() if self.metrics_history else time.time())
        }
    
    def get_metrics_history(self, hours: int = 1) -> List[PerformanceMetrics]:
        """Get metrics history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class HealthChecker:
    """Health check system"""
    
    def __init__(self, mann_model, monitor: MANNMonitor = None):
        self.mann_model = mann_model
        self.monitor = monitor
        self.logger = logging.getLogger("HealthChecker")
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check MANN model
        try:
            stats = self.mann_model.get_memory_statistics()
            health_status["checks"]["mann_model"] = {
                "status": "healthy",
                "memory_count": stats.get("total_memories", 0),
                "memory_utilization": stats.get("memory_utilization", 0.0)
            }
        except Exception as e:
            health_status["checks"]["mann_model"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        # Check system resources
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            health_status["checks"]["system_resources"] = {
                "status": "healthy",
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage
            }
            
            # Mark as unhealthy if resources are too high
            if cpu_usage > 90 or memory_usage > 90 or disk_usage > 95:
                health_status["checks"]["system_resources"]["status"] = "unhealthy"
                health_status["status"] = "unhealthy"
                
        except Exception as e:
            health_status["checks"]["system_resources"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "unhealthy"
        
        # Check monitoring system
        if self.monitor:
            try:
                perf_stats = self.monitor.get_performance_stats()
                health_status["checks"]["monitoring"] = {
                    "status": "healthy",
                    "performance_stats": perf_stats
                }
            except Exception as e:
                health_status["checks"]["monitoring"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return health_status
