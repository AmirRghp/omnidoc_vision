class OmniDocError(Exception):
    """Base exception for OmniDoc-Vision."""


class ConfigurationError(OmniDocError):
    """Invalid or missing application configuration."""


class PdfValidationError(OmniDocError):
    """The input PDF violates size, page, or format limits."""


class DocumentProcessingError(OmniDocError):
    """The document could not be parsed or transformed."""


class AssetAccessError(OmniDocError):
    """A crop or other document asset cannot be accessed safely."""


class IndexingError(OmniDocError):
    """A chunk could not be written to the dense or sparse index."""


class RetrievalError(OmniDocError):
    """Dense, sparse, or reranking retrieval failed."""


class GenerationError(OmniDocError):
    """The text or vision-language generation request failed."""
