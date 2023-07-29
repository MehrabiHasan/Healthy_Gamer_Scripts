

def get_results(query: str):
    """ 
    Send a query 
    """
    import requests  
    headers = {
    'accept': 'application/json',
    }

    params = {
        'terms': query,
    }

    response = requests.get('http://fastapi:8000/hgg/semantic_search', params=params, headers=headers)
    return response

def parse_results(data): 
    texts = [i.get("text") for i in data]
    titles = [i.get("title") for i in data]
    thumbnails = [i.get("thumbnail") for i in data]
    webps = [i.get("url") for i in data]
    return texts, titles, thumbnails, webps

def parse_webp(webp_img): 
    import requests # $ pip install requests
    from PIL import Image # $ pip install pillow
    import base64
    import io
    response = requests.get(webp_img)
    image_content = io.BytesIO(response.content)
    return Image.open(image_content)



def generate_unique_string():
    import uuid
    return str(uuid.uuid4())