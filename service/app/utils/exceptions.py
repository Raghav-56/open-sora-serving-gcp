"""
Custom exceptions for Open-Sora v2 API service.
"""


class OpenSoraServiceError(Exception):
    """Base exception for Open-Sora service."""
    pass


class ModelNotReadyError(OpenSoraServiceError):
    """Model is not ready for inference."""
    pass


class JobNotFoundError(OpenSoraServiceError):
    """Job ID not found."""
    pass


class VideoGenerationError(OpenSoraServiceError):
    """Video generation failed."""
    pass


class GCSUploadError(OpenSoraServiceError):
    """GCS upload operation failed."""
    pass


class WeightDownloadError(OpenSoraServiceError):
    """Model weight download failed."""
    pass
