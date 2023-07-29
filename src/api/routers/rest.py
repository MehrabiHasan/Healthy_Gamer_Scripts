from elasticsearch import AsyncElasticsearch
from fastapi import APIRouter, HTTPException, Query, Request
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from functools import lru_cache

from schemas.rest import (
    FullTextSearch,
    Semantic
)

# --- Helpers --- 
class Tokenizer(object):
    
    def __init__(self):
        self.devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device=self.devices)

    def get_token(self, documents):
        sentences  = [documents]
        sentence_embeddings = self.model.encode(sentences)
        encod_np_array = np.array(sentence_embeddings)
        encod_list = encod_np_array.tolist()
        return encod_list[0]
    
def get_url_from_thumbnail(thumbnail: str):
    from pathlib import Path
    p = Path(thumbnail)
    return p.parents[0].stem

def form_url(url: str, start: float):
    """
    This will form a youtube url with a start time
    """
    return f"https://www.youtube.com/watch?v={url}&t={start}s"

@lru_cache
def get_tokenizer():
   return Tokenizer()

router = APIRouter()

# --- Routes ---


@router.get(
    "/search",
    response_model=list[FullTextSearch],
    response_description="Search transcriptions by text, title",
)
async def search_by_keywords(
    request: Request,
    terms: str = Query(description="Search transcriptions by keywords in text, title"),
) -> list[Semantic] | None:
    result = await _search_by_keywords(request.app.client, terms)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No text or title with the provided terms '{terms}' found in database - please try again",
        )
    return result

@router.get(
    "/semantic_search",
    response_model=list[Semantic],
    response_description="Semantic Search through transcriptions by text, title",
)
async def knn_search(
    request: Request,
    terms: str = Query(description="Search transcriptions by Semantic Search in text, title"),
) -> list[Semantic] | None:
    result = await _knn_search(request.app.client, terms)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No text or title with the provided terms '{terms}' found in database - please try again",
        )
    return result

# --- Elasticsearch query funcs ---

async def _search_by_keywords(
    client: AsyncElasticsearch, terms: str
) -> list[FullTextSearch] | None:
    response = await client.search(
        index="healthygamergg-1",
        size=10,
        query={
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": terms,
                            "fields": ["text", "title"],
                            "minimum_should_match": 2,
                            "fuzziness": "AUTO",
                        }
                    }
                ],
            }
        }
    )
    result = response["hits"].get("hits")
    if result:
        data = []
        for item in result:
            data_dict = item["_source"]
            data.append(data_dict)
        return data
    return None

async def _knn_search(
    client: AsyncElasticsearch, terms: str
) -> list[Semantic] | None:
    #tokenize response 
    token_instance = get_tokenizer()
    token_vector = token_instance.get_token(terms)
    
    query = {
        "field": "embedding",
        "query_vector": token_vector,
        "k": 20,
        "num_candidates": 50
    }
    response = await client.knn_search(index="healthygamergg-1", knn=query, source=["text", "title","thumbnail","start"])
    result = response["hits"].get("hits")
    if result:
        data = []
        for item in result:
            data_dict = item["_source"]
            data_dict_new = {"text": data_dict['text'],
                             "title": data_dict['title'],
                             "thumbnail": data_dict['thumbnail'],
                             "url": form_url(url=get_url_from_thumbnail(data_dict.get('thumbnail')), 
                                             start=data_dict.get('start'))}
            data.append(data_dict_new)
        return data
    return None