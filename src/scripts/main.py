
import os 
import sys 
import pathlib
from pathlib import Path 
import json 
import glob
from elasticsearch import helpers
from dotenv import load_dotenv
from typing import *
from utils import * 
from argparse import ArgumentParser, Namespace

sys.path.insert(1, os.path.realpath(Path(__file__).resolve().parents[1]))

def grab_urls(yt_channel: str) -> None:
    """
    Start Method Grab all Youtube URls form channel 
    """
    try:
        if yt_channel != None:
            raw_urls = get_channel_urls(yt_channel)
            save_channel_urls(raw_urls)
    except Exception as E: 
        print(f"Error {E} w/ Traceback {E.__traceback__} ")

def get_video_info() -> None:
    """
    From saved urls, go and download all the audio 
    """
    try:
        urls = urls_text_to_list()
        pydict = get_audios(urls)
        save_channel_video_info(pydict)
    except Exception as E: 
        print(f"Error {E} w/ Traceback {E.__traceback__} ")

def transcribe_audio() -> None:
    """
    Take audio and transcribe videos with
    """
    try:
        whisper_model = get_whisper_model()
        whisper_transcribe(whisper_model)
    except Exception as E: 
        print(f"Error {E} w/ Traceback {E.__traceback__} ")

def create_embeddings() -> None:
    """
    Create embeddings dataframe from folder of transcriptions
    """
    try:
        form_embeddings()
    except Exception as E: 
        print(f"Error {E} w/ Traceback {E.__traceback__} ")

def ingest_to_db() -> None:
    """
    Create a list of iterators and ingest into elasticsearch with mapping
    """
    try:
        values = get_data() #list of iterators get iterator
        load_dotenv()
        INDEX_ALIAS = get_settings().elastic_index_alias
        assert INDEX_ALIAS
        if values:
            for item in values:
                ingest_iter(list(item)[0])
    except Exception as E: 
        print(f"Error {E} w/ Traceback {E.__traceback__} ")

def runner(yt_channel: str):
    """
    Main Runner
    """
    FUNC_MAPPER = {
        "grab_urls": grab_urls,
        "get_video_info": get_video_info,
        "transcribe_audio": transcribe_audio,
        "create_embeddings": create_embeddings,
        "ingest_to_db": ingest_to_db
    }
    for function_name, func in FUNC_MAPPER.items():
        print(f"Running: {function_name}")
        if function_name == "grab_urls":
            func(yt_channel=yt_channel)
        else:
            func()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('yt_channel',help="This is a youtube channel url", type=str)

    args: Namespace = parser.parse_args()

    runner(yt_channel=args.yt_channel)

