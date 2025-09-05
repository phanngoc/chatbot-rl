"""
MANN API Server và Client
RESTful API cho MANN system
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from contextlib import asynccontextmanager

from .mann_core import MemoryAugmentedNetwork, MemoryBankEntry
from .mann_config import MANNConfig
from .mann_monitoring import MANNMonitor, PagerSystem


# Pydantic models
class MemoryRequest(BaseModel):
    content: str = Field(..., description="Memory content")
    context: str = Field("", description="Context information")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    importance_weight: float = Field(1.0, ge=0.0, le=3.0, description="Importance weight")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MemoryUpdateRequest(BaseModel):
    memory_id: str = Field(..., description="Memory ID to update")
    content: Optional[str] = Field(None, description="New content")
    context: Optional[str] = Field(None, description="New context")
    importance_weight: Optional[float] = Field(None, ge=0.0, le=3.0, description="New importance weight")
    tags: Optional[List[str]] = Field(None, description="New tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="New metadata")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results")
    min_similarity: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")


class QueryRequest(BaseModel):
    input_text: str = Field(..., description="Input text for processing")
    retrieve_memories: bool = Field(True, description="Whether to retrieve memories")


class MemoryResponse(BaseModel):
    memory_id: str
    content: str
    context: str
    importance_weight: float
    usage_count: int
    tags: List[str]
    timestamp: str
    similarity: Optional[float] = None
    attention_weight: Optional[float] = None


class QueryResponse(BaseModel):
    output: str
    memory_info: List[MemoryResponse]
    processing_time: float
    memory_utilization: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    memory_count: int
    memory_utilization: float
    uptime: float
    version: str


# Global variables
mann_model: Optional[MemoryAugmentedNetwork] = None
config: Optional[MANNConfig] = None
monitor: Optional[MANNMonitor] = None
pager: Optional[PagerSystem] = None
start_time: datetime = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global mann_model, config, monitor, pager, start_time
    
    # Startup
    start_time = datetime.now()
    config = MANNConfig()
    config.update_from_env()
    
    # Initialize MANN model
    mann_model = MemoryAugmentedNetwork(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        memory_size=config.memory_size,
        memory_dim=config.memory_dim,
        output_size=config.output_size,
        device=torch.device('cpu')
    )
    
    # Load existing memory bank
    if os.path.exists(config.memory_save_path):
        mann_model.load_memory_bank(config.memory_save_path)
    
    # Initialize monitoring
    if config.enable_monitoring:
        monitor = MANNMonitor(config)
        pager = PagerSystem(config.pager_webhook_url) if config.enable_pager else None
    
    logging.info("MANN API server started")
    yield
    
    # Shutdown
    if mann_model:
        mann_model.save_memory_bank(config.memory_save_path)
    logging.info("MANN API server stopped")


# FastAPI app
app = FastAPI(
    title="MANN API",
    description="Memory-Augmented Neural Network API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    uptime = (datetime.now() - start_time).total_seconds()
    stats = mann_model.get_memory_statistics()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        memory_count=stats.get("total_memories", 0),
        memory_utilization=stats.get("memory_utilization", 0.0),
        uptime=uptime,
        version="1.0.0"
    )


@app.post("/memories", response_model=Dict[str, str])
async def add_memory(request: MemoryRequest):
    """Add new memory"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    try:
        memory_id = mann_model.add_memory(
            content=request.content,
            context=request.context,
            tags=request.tags,
            importance_weight=request.importance_weight,
            metadata=request.metadata
        )
        
        # Save memory bank
        mann_model.save_memory_bank(config.memory_save_path)
        
        return {"memory_id": memory_id, "status": "created"}
    
    except Exception as e:
        logging.error(f"Failed to add memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}")
async def update_memory(memory_id: str, request: MemoryUpdateRequest):
    """Update existing memory"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    try:
        success = mann_model.update_memory(
            memory_id=memory_id,
            content=request.content,
            context=request.context,
            importance_weight=request.importance_weight,
            tags=request.tags,
            metadata=request.metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Save memory bank
        mann_model.save_memory_bank(config.memory_save_path)
        
        return {"status": "updated"}
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete memory"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    try:
        success = mann_model.delete_memory(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        # Save memory bank
        mann_model.save_memory_bank(config.memory_save_path)
        
        return {"status": "deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[MemoryResponse])
async def search_memories(request: SearchRequest):
    """Search memories"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    try:
        results = mann_model.search_memories(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        return [
            MemoryResponse(
                memory_id=result["id"],
                content=result["content"],
                context=result["context"],
                importance_weight=result["importance_weight"],
                usage_count=result["usage_count"],
                tags=result["tags"],
                timestamp=result["timestamp"],
                similarity=result["similarity"]
            )
            for result in results
        ]
    
    except Exception as e:
        logging.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process query with MANN"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    try:
        start_time = datetime.now()
        
        # Convert text to tensor (simplified - in production use proper embeddings)
        input_tensor = torch.randn(1, 1, config.input_size)  # Dummy input
        
        # Process with MANN
        output, memory_info = mann_model.forward(input_tensor, request.retrieve_memories)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert memory info to response format
        memory_responses = [
            MemoryResponse(
                memory_id=info["id"],
                content=info["content"],
                context=info["context"],
                importance_weight=info["importance_weight"],
                usage_count=info["usage_count"],
                tags=info["tags"],
                timestamp=datetime.now().isoformat(),
                similarity=info.get("similarity"),
                attention_weight=info.get("attention_weight")
            )
            for info in memory_info
        ]
        
        # Background monitoring
        if monitor:
            background_tasks.add_task(monitor.record_query, processing_time, len(memory_info))
        
        return QueryResponse(
            output="Processed successfully",  # Simplified output
            memory_info=memory_responses,
            processing_time=processing_time,
            memory_utilization=mann_model.get_memory_statistics().get("memory_utilization", 0.0)
        )
    
    except Exception as e:
        logging.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get MANN statistics"""
    if not mann_model:
        raise HTTPException(status_code=503, detail="MANN model not initialized")
    
    try:
        stats = mann_model.get_memory_statistics()
        return stats
    
    except Exception as e:
        logging.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Client class
class MANNClient:
    """Client for MANN API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = None
    
    async def __aenter__(self):
        import aiohttp
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def add_memory(self, content: str, context: str = "", tags: List[str] = None, 
                        importance_weight: float = 1.0, metadata: Dict[str, Any] = None) -> str:
        """Add memory"""
        data = {
            "content": content,
            "context": context,
            "tags": tags or [],
            "importance_weight": importance_weight,
            "metadata": metadata or {}
        }
        
        async with self.session.post(f"{self.base_url}/memories", json=data) as response:
            result = await response.json()
            return result["memory_id"]
    
    async def search_memories(self, query: str, top_k: int = 5, 
                            min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Search memories"""
        data = {
            "query": query,
            "top_k": top_k,
            "min_similarity": min_similarity
        }
        
        async with self.session.post(f"{self.base_url}/search", json=data) as response:
            return await response.json()
    
    async def process_query(self, input_text: str, retrieve_memories: bool = True) -> Dict[str, Any]:
        """Process query"""
        data = {
            "input_text": input_text,
            "retrieve_memories": retrieve_memories
        }
        
        async with self.session.post(f"{self.base_url}/query", json=data) as response:
            return await response.json()


def run_server(config: MANNConfig = None):
    """Run MANN API server"""
    if config is None:
        config = MANNConfig()
    
    uvicorn.run(
        "mann_api:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    import os
    config = MANNConfig()
    config.update_from_env()
    run_server(config)
