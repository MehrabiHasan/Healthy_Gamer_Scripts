import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI


from config import Settings
from routers import rest

@lru_cache()
def get_settings():
    # Use lru_cache to avoid loading .env file for every request
    return Settings()

# Connect to ElasticSearch
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for Elasticsearch connection."""
    settings = get_settings()
    username = settings.elastic_user
    password = settings.elastic_password
    port = settings.elastic_port
    ELASTIC_URL = settings.elastic_url
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elastic_client = AsyncElasticsearch(
            f"http://{ELASTIC_URL}:{port}",
            basic_auth=(username, password),
            request_timeout=60,
            max_retries=3,
            retry_on_timeout=True,
            verify_certs=False,
        )
        """Async context manager for Elasticsearch connection."""
        app.client = elastic_client
        print("Successfully connected to Elasticsearch")
        yield
        await elastic_client.close()
        print("Successfully closed Elasticsearch connection")

app = FastAPI(
    title="REST API for wine reviews on Elasticsearch",
    description=(
        "Query from an Elasticsearch database of Healthygamergg Transcriptions"
    ),
    version=get_settings().tag,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "REST API for querying Elasticsearch database of Healthygamergg transcriptions: https://www.youtube.com/@HealthyGamerGG"
    }


# Attach routes
app.include_router(rest.router, prefix="/hgg", tags=["hgg"])