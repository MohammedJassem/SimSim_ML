from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from transformers import pipeline
from typing import List 

app = FastAPI()


ner_model = pipeline('ner', model='dslim/bert-base-NER', device=-1)  

class SentenceRequest(BaseModel):
    sentence: str

class EntityResponse(BaseModel):
    word: str
    score: float
    entity_group: str
    start: int
    end: int

@app.post('/extract_entities', response_model=List[EntityResponse])
async def extract_entities(request_data: SentenceRequest = Body(...)):
    try:
        sentence = request_data.sentence

       
        entities = ner_model(sentence)

        result = [
            EntityResponse(
                word=entity['word'],
                score=entity['score'],
                entity_group=entity['entity_group'],
                start=entity['start'],
                end=entity['end']
            )
            for entity in entities
        ]

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', debug=True)
