"""
Agent Communication System
Handles inter-agent messaging and coordination
"""

from __future__ import annotations

import asyncio
import threading
import queue
import logging
from typing import Dict, List, Optional, Callable
from collections import defaultdict
from datetime import datetime
import uuid

from .base_agent import  DynamicAgent, LearningSignal as AgentMessage

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Central message broker for agent communication
    Supports synchronous and asynchronous messaging
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize broker state"""
        self.agents: Dict[str, DynamicAgent] = {}
        self.message_queues: Dict[str, queue.Queue] = defaultdict(queue.Queue)
        self.response_queues: Dict[str, Dict[str, queue.Queue]] = defaultdict(dict)
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)
        self.message_history: List[Dict] = []
        self.running = True
        
        # Start message processor thread
        self.processor_thread = threading.Thread(target=self._process_messages, daemon=True)
        self.processor_thread.start()
        
        logger.info("MessageBroker initialized")
    
    def register_agent(self, agent: DynamicAgent):
        """Register an agent with the broker"""
        self.agents[agent.name] = agent
        logger.info(f"Agent '{agent.name}' registered with broker")
    
    def send(self, message: AgentMessage, timeout: float = 5.0) -> Optional[Dict]:
        """
        Send a message and optionally wait for response
        """
        # Add to sender's queue for processing
        if message.sender not in self.message_queues:
            self.message_queues[message.sender] = queue.Queue()
        
        # Store in history
        self.message_history.append({
            **message.to_dict(),
            "status": "sent",
            "timestamp": datetime.now().isoformat()
        })
        
        # Route to receiver
        if message.receiver == "broadcast":
            self._broadcast(message)
        elif message.receiver in self.agents:
            self._deliver_to_agent(message)
        else:
            logger.error(f"Unknown receiver: {message.receiver}")
            return None
        
        # Wait for response if required
        if message.requires_response:
            response_queue = self.response_queues[message.sender][message.correlation_id]
            try:
                response = response_queue.get(timeout=timeout)
                return response
            except queue.Empty:
                logger.warning(f"Timeout waiting for response to {message.correlation_id}")
                return None
        
        return {"status": "sent", "message_id": message.id}
    
    def _deliver_to_agent(self, message: AgentMessage):
        """Deliver message to specific agent"""
        agent = self.agents[message.receiver]
        
        # Process in separate thread to avoid blocking
        def deliver():
            response = agent.receive_message(message)
            if response and message.requires_response:
                self._send_response(message, response)
        
        thread = threading.Thread(target=deliver, daemon=True)
        thread.start()
    
    def _broadcast(self, message: AgentMessage):
        """Broadcast to all agents or subscribers"""
        for agent_name, agent in self.agents.items():
            if agent_name != message.sender:  # Don't send to self
                broadcast_msg = AgentMessage(
                    sender=message.sender,
                    receiver=agent_name,
                    message_type=message.message_type,
                    payload=message.payload,
                    correlation_id=message.correlation_id
                )
                self._deliver_to_agent(broadcast_msg)
    
    def _send_response(self, original_message: AgentMessage, response_data: Dict):
        """Send response to original sender"""
        response = AgentMessage(
            sender=original_message.receiver,
            receiver=original_message.sender,
            message_type="response",
            payload=response_data,
            correlation_id=original_message.correlation_id,
            requires_response=False
        )
        
        # Store in response queue for the original sender
        if original_message.sender in self.response_queues:
            if original_message.correlation_id in self.response_queues[original_message.sender]:
                self.response_queues[original_message.sender][original_message.correlation_id].put(response_data)
    
    def _process_messages(self):
        """Background thread to process message queues"""
        while self.running:
            # Process each agent's queue
            for agent_name, msg_queue in list(self.message_queues.items()):
                try:
                    while not msg_queue.empty():
                        message = msg_queue.get_nowait()
                        self._deliver_to_agent(message)
                except queue.Empty:
                    continue
            
            import time
            time.sleep(0.1)  # Small delay to prevent CPU spinning
    
    def get_message_history(self, agent_name: Optional[str] = None) -> List[Dict]:
        """Get message history for debugging"""
        if agent_name:
            return [m for m in self.message_history if m["sender"] == agent_name or m["receiver"] == agent_name]
        return self.message_history
    
    def shutdown(self):
        """Shutdown the broker"""
        self.running = False
        logger.info("MessageBroker shutting down")


class AgentConversation:
    """
    Manages multi-turn conversations between agents
    """
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.conversations: Dict[str, List[Dict]] = {}
    
    def start_conversation(self, conversation_id: str, initiator: str, topic: str) -> str:
        """Start a new conversation between agents"""
        self.conversations[conversation_id] = [{
            "id": conversation_id,
            "initiator": initiator,
            "topic": topic,
            "messages": [],
            "timestamp": datetime.now().isoformat(),
            "status": "active"
        }]
        return conversation_id
    
    def add_message(self, conversation_id: str, message: AgentMessage):
        """Add a message to a conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id][0]["messages"].append({
                "from": message.sender,
                "to": message.receiver,
                "type": message.message_type,
                "payload": message.payload,
                "timestamp": message.timestamp
            })
    
    def get_conversation(self, conversation_id: str) -> Optional[List[Dict]]:
        """Get full conversation history"""
        return self.conversations.get(conversation_id)
    
    def close_conversation(self, conversation_id: str, outcome: str):
        """Close a conversation with outcome"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id][0]["status"] = "closed"
            self.conversations[conversation_id][0]["outcome"] = outcome
            self.conversations[conversation_id][0]["closed_at"] = datetime.now().isoformat()
