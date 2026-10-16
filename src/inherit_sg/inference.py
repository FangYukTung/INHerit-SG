"""Shared query setup for the command line and benchmark adapters."""
from .query import QueryParser, parse_object
from .retrieval import Memory, RetrievalSettings, StructuredRetriever
from .visual import VisualReranker


def query_components(config, *, parse=True, visual=True):
    parser = reranker = router = None
    tasks = []
    if parse: tasks.append('query_parsing')
    if visual: tasks.append('visual_reranking')
    if tasks:
        from .model_router import ModelRouter
        settings = dict(config['llm_config']['model_router'])
        settings['routing'] = {task: settings['routing'][task] for task in tasks}
        router = ModelRouter(settings)
    if parse:
        parser = QueryParser(*router.resolve('query_parsing'))
    if visual:
        reranker = VisualReranker(*router.resolve('visual_reranking'), **config.get('visual_reranking', {}))
    return parser, reranker


def retriever_for(directory, config):
    return StructuredRetriever(Memory.load(directory), settings=RetrievalSettings(**config.get('retrieval', {})))
