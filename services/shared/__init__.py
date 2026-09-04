"""Shared utilities and models for RAG application services.

Imports are lazy to avoid pulling in heavy dependencies (sqlalchemy, torch, etc.)
for services that only need lightweight modules like logging or metrics.
"""


def __getattr__(name):
    """Lazy import shared modules on first access."""
    _imports = {
        # models.py
        "Document": (".models", "Document"),
        "TextChunk": (".models", "TextChunk"),
        "Conversation": (".models", "Conversation"),
        "Message": (".models", "Message"),
        "Source": (".models", "Source"),
        "IngestionJob": (".models", "IngestionJob"),
        # database.py
        "DatabaseManager": (".database", "DatabaseManager"),
        "get_db_manager": (".database", "get_db_manager"),
        "init_db": (".database", "init_db"),
        "get_session": (".database", "get_session"),
        # config.py
        "Settings": (".config", "Settings"),
        "get_settings": (".config", "get_settings"),
        # vector_search_service.py
        "VectorSearchService": (".vector_search_service", "VectorSearchService"),
        # llm_guard_service.py
        "LLMGuardService": (".llm_guard_service", "LLMGuardService"),
        # orm_models.py
        "Base": (".orm_models", "Base"),
        "DocumentORM": (".orm_models", "Document"),
        "TextChunkORM": (".orm_models", "TextChunk"),
        "ConversationORM": (".orm_models", "Conversation"),
        "MessageORM": (".orm_models", "Message"),
        "IngestionJobORM": (".orm_models", "IngestionJob"),
    }

    if name in _imports:
        module_path, attr_name = _imports[name]
        import importlib
        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Document",
    "TextChunk",
    "Conversation",
    "Message",
    "Source",
    "IngestionJob",
    "DatabaseManager",
    "get_db_manager",
    "init_db",
    "get_session",
    "Settings",
    "get_settings",
    "VectorSearchService",
    "LLMGuardService",
    "Base",
    "DocumentORM",
    "TextChunkORM",
    "ConversationORM",
    "MessageORM",
    "IngestionJobORM",
]
