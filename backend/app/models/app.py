from fastapi import FastAPI
from inference import LegalSummarizer
import uvicorn

app = FastAPI()
summarizer = LegalSummarizer()

@app.post("/summarize")
async def summarize(text: str):
    result = summarizer.summarize(text)
    return {
        "summary": result["summary"],
        "citations": result["citations"],
        "entities": result["entities"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)