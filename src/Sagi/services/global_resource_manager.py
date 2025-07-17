import logging
from typing import Optional

from Sagi.services.model_client_service import ModelClientService


class GlobalResourceManager:
    """
    Global resource manager for centralized service management.
    
    This manager provides singleton access to core services like ModelClientService,
    ensuring consistent resource management across the application.
    """
    
    _instance: Optional['GlobalResourceManager'] = None
    _model_client_service: Optional[ModelClientService] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logging.info("🌐 GlobalResourceManager created")
        return cls._instance
    
    @classmethod
    def get_model_client_service(cls) -> ModelClientService:
        """
        Get the singleton ModelClientService instance.
        
        Returns:
            ModelClientService: The singleton service instance
        """
        if cls._model_client_service is None:
            cls._model_client_service = ModelClientService()
            logging.info("🔧 ModelClientService singleton created")
        return cls._model_client_service
    
    @classmethod
    def reset(cls):
        """Reset all services - useful for testing."""
        if cls._model_client_service:
            cls._model_client_service.clear_cache()
        cls._model_client_service = None
        cls._instance = None
        logging.info("🔄 GlobalResourceManager reset")
    
    @classmethod
    def get_service_status(cls) -> dict:
        """Get status of all managed services."""
        status = {
            "model_client_service": cls._model_client_service is not None
        }
        
        if cls._model_client_service:
            status["model_client_cache"] = cls._model_client_service.get_cache_info()
        
        return status