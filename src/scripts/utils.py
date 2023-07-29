
import os
import shutil
import pathlib
from tqdm import tqdm
import json
import subprocess
from typing import List
import torch
import pandas as pd 
import numpy as np
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import glob 
from elasticsearch import Elasticsearch
# -------------------------------------------- yt-dlp --------------------------------------------
def get_channel_urls(yt_channel: str):
    """
    Given a youtube channel go throught a collect all the video urls
    """
    proc = subprocess.Popen(['yt-dlp','--flat-playlist', '--print', 'id',yt_channel], stdout=subprocess.PIPE)
    output = proc.stdout.read()
    var = output.decode().split('\n')[:-1]
    return_list = [f'https://www.youtube.com/watch?v={v}' for v in var]
    return return_list

def urls_text_to_list(path: str='/home/myuser/code/src/data/urls.txt'):
    """
    Read in a text file of urls
    """
    with open(path, 'r') as p:
        files = p.read().splitlines()
    return files

def save_channel_urls(return_lst: list): 
    """
    Given a list of channel urls - save to file 
    """
    with open('/home/myuser/code/src/data/urls.txt','w') as fp:
        for item in tqdm(return_lst, desc="Processing Channel Urls"):
            fp.write(f'{item}\n')
        fp.close()

def save_channel_video_info(results_dict: dict): 
    """
    Given a dictionary of video info - save it to a json object
    """
    with open('/home/myuser/code/src/data/vids.json','w') as fp:
        json.dump(results_dict, fp)
        print('saved json')

class str2(str):
  def __repr__(self):
    return ''.join(('"', super().__repr__()[1:-1], '"'))


def get_audios(urls: List[str]):
  """
  Given a list of urls, go through and download the audio. Also grab the video thumbnail. 
  """
  import yt_dlp
  import subprocess

  paths = {}
  filtered_list = [item for item in urls if item != ""]
  pbar = tqdm(filtered_list)
  temp_dir = '/home/myuser/code/src/data/vids'
  
  ydl = yt_dlp.YoutubeDL({
        'quiet': True,
        'verbose': False,
        'ignoreerrors': True,
        'format': 'bestaudio',
        "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
        'postprocessors': [{'preferredcodec': 'mp3', 'preferredquality': '192', 'key': 'FFmpegExtractAudio', }],
                        })
    
  for url in pbar:
      result = ydl.extract_info(url=url, download=True)
      pbar.set_description(f"Processing: {result['title']}")
      paths[result["title"]] = [os.path.join(temp_dir, f"{result['id']}.mp3"), result['thumbnail'] ]
  return paths
  
# -------------------------------------------- yt-dlp --------------------------------------------
# -------------------------------------------- Faster Whisper --------------------------------------------
def get_whisper_model(model_size: str="tiny"):
    """
    Opt for whisper with our own VAD, this is useful exercise for other audio models that don't necessarily 
    Have VADs built in like Meta's MMS
    """
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    return whisper_model 

def whisper_transcribe(whisper_model,save_path='/home/myuser/code/src/data/transcriptions',working_dir = '/home/myuser/code/src/data/vids'):
  """
    Go Through Each Transcriptions and Transcribe with whisper model
  """
  
  import os
  import pathlib
  import json
  vids = glob.glob(f"{working_dir}/*.mp3")
  for loc, wav_file in tqdm(enumerate(vids)):
    print(f"Working on {wav_file}")
    filename = pathlib.Path(wav_file).stem
    segments, info = whisper_model.transcribe(wav_file, beam_size=5)
    segment_list = [i for i in segments]
    save_list = []
    for r in segment_list:
        d = r._asdict()
        del d["tokens"]
        save_list.append(d)
    os.makedirs(save_path, exist_ok=True)
    save_json = os.path.join(save_path,filename + '.json')
    with open(save_json,'w') as jf:
      json.dump(save_list,jf)

# -------------------------------------------- Faster Whisper --------------------------------------------
# -------------------------------------------- Embeddings --------------------------------------------
def get_embedding_model(model: str='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'):
  """
  Download Embedding Model
  """
  from sentence_transformers import SentenceTransformer
  devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
  model = SentenceTransformer(model, device=devices)
  return model 
   
def get_orig_file_path(file_path):
  """
  Mainly Used if VAD chunks was used
  """
  temp1 = file_path.split('.')[0] 
  if len(temp1) == 11:
    word = temp1 
  else:
    word = temp1.rpartition('_')[0]
  return word

def form_dataframe(json_path):
  """
  Create Dataframe
  """
  import pandas as pd
  import json
  with open(json_path, 'r') as jf:
    data = json.load(jf)
  df = pd.DataFrame(data)
  return df

def get_vids_json(path: str = '/home/myuser/code/src/data/vids.json'):
  """
  Get Json 
  """
  with open(path,'r') as jf:
    vids = json.load(jf)
  return vids

def get_keys_from_value(d, val):
    
    return [k for k, v in d.items() if v == val]

def get_more_metrics(df, vids, filenames, loc):
  """
  Add Additional Info to Data Frame
  """
  orig_file = filenames[loc]
  temp = list(vids.values())
  f_names = [pathlib.Path(i[0]).stem for i in temp]
  f_ind = f_names.index(orig_file) ### Get file index
  vid_name = get_keys_from_value(vids,temp[f_ind])[0]
  vid_img = temp[f_ind][1]
  df['title'] = [vid_name]*len(df)
  df['thumbnail'] = [vid_img]*len(df)
  return df

def calculate_embedding(string, model):
  return model.encode(string)

def apply_embedding(df, model):
  """
  Go through text string and apply embedding function to it
  """
  from tqdm import tqdm
  tqdm.pandas(desc='My bar!')
  df['embedding'] = df['text'].progress_apply(lambda x: calculate_embedding(x, model))
  return df 

def form_embeddings(transcript_dir = '/home/myuser/code/src/data/transcriptions'):
  """
  Form embeddings dataframe
  """
  import numpy as np 
  import pandas as pd
  from tqdm import tqdm
  embedding_path = '/home/myuser/code/src/data/embeddings'
  if os.path.exists != True:
    os.makedirs(embedding_path, exist_ok=True)
  files = os.listdir(transcript_dir)
  filenames = [get_orig_file_path(pathlib.Path(i).stem) for i in files]
  transcript_paths = [os.path.join(transcript_dir, i) for i in files]
  embed_model = get_embedding_model()
  vids = get_vids_json()
  for loc, path in tqdm(enumerate(transcript_paths)):
    json_fp = os.path.join(embedding_path,pathlib.Path(path).stem + '.json')
    with open(path,'r') as jf:
        data = json.load(jf)
    raw_df = pd.DataFrame(data)
    vid_df = get_more_metrics(raw_df, vids, filenames, loc) ### Could be weird with split video - maybe test on original video itself not my split
    embed_df = apply_embedding(vid_df, embed_model)
    embed_df['document_embedding'] = [np.mean(embed_df['embedding'])]*len(embed_df)
    embed_df['embedding'] = embed_df['embedding'].apply(lambda x: list(map(str,x))) ###TODO: Re float32 it when you ingest into Database
    embed_df['document_embedding'] = embed_df['document_embedding'].apply(lambda x: list(map(str,x)))
    result = embed_df.to_dict(orient='records')

    with open(json_fp,'w') as jf:
        json.dump(result, jf)

# -------------------------------------------- Embeddings --------------------------------------------
def final_fixer(json_str: str):
  """
  Go through all text in embeddings and fix it if necessary
  """
  from tqdm import tqdm 
  import re
  get_all = re.findall('{(.+?)}',json_str) # Regex to Get all Text in between curly brackets
  items = []
  for loc_a, ind_dict in tqdm(enumerate(get_all)): ## loop through all dictionaries
    split_comma = ind_dict.split(':')
    start = 0 
    end = len(split_comma)-1
    keys = [] 
    values = []
    try:
      for loc,i in enumerate(split_comma):
        if loc == start: 
          i = i.replace('"','').strip()
          keys.append(i)
        elif loc == end:
          i = i.replace('"','').strip()
          values.append(i) 
        elif loc == 12:
          loc_ = i.rfind(',')
          val = i[:loc_]
          loc_2 = split_comma[loc+1].rfind(',')
          val2 = split_comma[loc+1][:loc_2]
          new_val = val + val2 
          new_key = split_comma[loc+1][loc_2+1:].replace('"','').strip()
          keys.append(new_key)
          values.append(new_val)
        elif loc == 13: 
          continue
        else:
          if loc <= 16:
            loc_ = i.rfind(',')
            value = i[:loc_].replace('"','').strip()
            key = i[loc_:].split('"')[1] 
            keys.append(key)
            values.append(value) 
    except Exception as E:
      print(f'{E} as {loc_a}:{loc}')
      print(split_comma[loc])
    items.append(dict(zip(keys,values)))
  return items

def parse_ndjson(data):
  """
  Get Json
  """
  return [json.loads(l) for l in data]

def parse_json(fp: str, basepath: str="/content"): 
  """
  Parse File to get Valid Response 
  """
  import os 
  full_path = os.path.join(basepath, fp)
  try: 
    try: # Case 1
      with open(full_path, encoding="utf-8") as jf: 
        data = json.load(jf)
        return data, 1
    except: #Case
      try: #case 2
        with open(full_path, encoding="utf-8-sig") as jf:
          dicts = parse_ndjson(jf.readlines())
          data = dicts[0]
          return data, 2
      except:
        try: #case 3: Hail mary
          with open(full_path, encoding="utf-8-sig") as jf:
            dd = jf.read()
            data = final_fixer(dd)
            return data, 3
        except:
          print("Sorry Can't open File")

  except Exception as E:
    print("Sorry couldn't Open File")

def clean_dataframe(raw_df): 
  """
  Go through and clean dataframe
  """
  import ast
  raw_df = raw_df.drop(['temperature', 'avg_logprob','compression_ratio', 'no_speech_prob', 'words'], axis=1)
  raw_df['start'] = raw_df['start'].apply(ast.literal_eval)
  raw_df['end'] = raw_df['end'].apply(ast.literal_eval)
  raw_df['embedding'] = raw_df['embedding'].apply(ast.literal_eval)
  raw_df['document_embedding'] = raw_df['document_embedding'].apply(ast.literal_eval)
  return raw_df

# -------------------------------------------- Ingest Database --------------------------------------------
"""
Inspired from: https://github.com/prrao87/async-db-fastapi/tree/main/dbs/elasticsearch 
"""
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch, helpers
from pathlib import Path 
from functools import lru_cache, partial
from typing import Any, Iterator
import srsly
import warnings
import asyncio
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(1, os.path.realpath(Path(__file__).resolve().parents[1]))
from api.config import Settings
from schemas.schema import PDValidator, DictValidator
from pydantic.main import ModelMetaclass

load_dotenv()

# Custom types
items = dict[str, Any]

@lru_cache
def get_settings():
   return Settings()

def validate_json(fp: str, fn: str):
    """
    Parse embedding.json into dataframe and validate
    """
    import json 
    ## OPen Beddings and Get Case
    data_, case_ = parse_json(fp)
    ## Turn Data into dataframe to validate
    df = pd.DataFrame(data_)
    if case_ == 3:
        df = clean_dataframe(df)
    # Validate
    df = PDValidator(df_dict=df.to_dict(orient='records')) ### Return PDValidator Object 
    ## Convert to Json 
    data = json.loads(df.json()) ### Convert to Json Object 
    ## Get Original Object
    main = data.get('df_dict') ### Get Original Data - what is this 
    # If this is a list of dict, then just global ids here and you should be good?
    ids = [f"{fn}_{l.get('id')}" for l in main]
    result = [dict(item, global_id=f"{ids[loc]}") for loc, item in tqdm(enumerate(main),total=len(main))]
    yield result
     
def get_data(data_dir='/home/myuser/code/src/data/embeddings'): # Returns list of iterators 
  """
  Go through Embeddings, validate data through pydantic and return as a list of dictionaries (each dictionary is an iterator)
  """
  files = os.listdir(data_dir)
  filenames = [get_orig_file_path(pathlib.Path(i).stem) for i in files]
  embedding_paths = [os.path.join(data_dir, i) for i in files]
  test_paths = embedding_paths
  pylist = [] #a list of iterators
  for loc, fp in tqdm(enumerate(test_paths),total=len(test_paths)): # Turn into an iterator
     x = validate_json(fp, filenames[loc]) #Now an Iterator
     pylist.append(x)
  return pylist

def create_sync_es(settings):
  USERNAME = settings.elastic_user
  PASSWORD = settings.elastic_password
  PORT = settings.elastic_port
  ELASTIC_URL = settings.elastic_url
  # Connect to ElasticSearch
  elastic_client = Elasticsearch(
      f"http://{ELASTIC_URL}:{PORT}",
      basic_auth=(USERNAME, PASSWORD),
      request_timeout=300,
      max_retries=3,
      retry_on_timeout=True,
      verify_certs=False,
  )
  return elastic_client

def get_results(doc, destination_index):
        action = {
            "_index": destination_index,
            "_id": doc["_id"],
            "_source": doc["_source"],
        }
        yield action 

def create_pre_index(es: Elasticsearch, path: Path) -> None:
  """
  Create An Elasticsearch index if it doesn't already exist
  """
  import elasticsearch
  elastic_config = dict(srsly.read_json(path))
  assert elastic_config is not None
  INDEX_ALIAS = get_settings().elastic_index_alias
  index_name = f"{INDEX_ALIAS}-1"

  if not es.indices.exists_alias(name=INDEX_ALIAS):
      print(f"Did not find index {INDEX_ALIAS} in db, creating index...\n")
      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          mappings = elastic_config.get("mappings")
          settings = elastic_config.get("settings")
          try:
              es.indices.create(index=index_name, mappings=mappings, settings=settings)
              es.indices.put_alias(index=index_name, name=INDEX_ALIAS)
          except elasticsearch.BadRequestError as ebr:
             print(f"{ebr} Issues: Most likely the Correct Index Already exists")
          except Exception as E:
             print(f"{E} Couldn't create index")
  
  return index_name

   
def update_documents(client: Elasticsearch,
                     index: str, 
                     data: list[items],
                     CHUNKSIZE: int=10000) -> None:
  def generate_doc():
     for doc in data:
        action = {
           "_index": index,
           "_source": doc
        }
        yield action 
  try:
    for success, info in helpers.parallel_bulk(
      client=client,
      actions=generate_doc(),
      thread_count = 10,
      chunk_size=CHUNKSIZE # Use Maximum Size of Embeddings
    ):
      if not success:
          print(f"Failed to Index docs: {info}")
  except helpers.BulkIndexError as hbe:
     print(f"Failed to Index {CHUNKSIZE} Documents: {hbe}")
  except Exception as E:
     print(f"Something went wrong")

def ingest_iter(data: list[items]) -> None:
  settings = get_settings()
  try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        elastic_client = create_sync_es(settings)
        index_name = create_pre_index(elastic_client, Path("/home/myuser/code/src/scripts/mappings/mappings.json"))
        update_documents(client=elastic_client,
                         index=index_name,
                         data = data)
        elastic_client.close()
  except Exception as E:
    elastic_client.close()
    print(f"Exited w/ Error: {E}")

# -------------------------------------------- Ingest Database --------------------------------------------