from pathlib import Path
import os
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import models as qdrant_models

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI()

_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")
_OPENAI = OpenAI()
_PAYLOAD_INDEX_READY: set[str] = set()


def _qdrant_url() -> str:
    url = os.getenv("QDRANT_DB_URL")
    if not url:
        raise HTTPException(status_code=500, detail="QDRANT_DB_URL is not set")
    return url


def _qdrant_api_key() -> str | None:
    return os.getenv("QDRANT_API_KEY")


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=_qdrant_url(), api_key=_qdrant_api_key())


def _ensure_doc_id_payload_index(collection_name: str) -> None:
    if collection_name in _PAYLOAD_INDEX_READY:
        return
    client = _get_qdrant_client()
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.doc_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except UnexpectedResponse as e:
        if getattr(e, "status_code", None) != 409:
            raise HTTPException(
                status_code=500,
                detail="Failed to create Qdrant payload index for metadata.doc_id",
            ) from e
    _PAYLOAD_INDEX_READY.add(collection_name)


def _split_documents(documents, chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents=documents)


def index_file(
    file_path: Path,
    *,
    collection_name: str,
    doc_id: str | None = None,
    original_filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 400,
    batch_size: int = 16,
    timeout: int = 120,
):
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(file_path=file_path)
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(file_path=str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    pages = loader.load()
    if doc_id:
        for page in pages:
            if isinstance(getattr(page, "metadata", None), dict):
                page.metadata["doc_id"] = doc_id
                if original_filename:
                    page.metadata["original_filename"] = original_filename
    chunks = _split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if doc_id:
        for chunk in chunks:
            if isinstance(getattr(chunk, "metadata", None), dict):
                chunk.metadata["doc_id"] = doc_id
                if original_filename:
                    chunk.metadata["original_filename"] = original_filename

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=_EMBEDDINGS,
        url=_qdrant_url(),
        api_key=_qdrant_api_key(),
        collection_name=collection_name,
        batch_size=batch_size,
        timeout=timeout,
    )

    return {
        "collection_name": collection_name,
        "file": file_path.name,
        "pages": len(pages),
        "chunks": len(chunks),
        "vector_store": vector_store,
    }


def _get_vector_store(collection_name: str):
    try:
        return QdrantVectorStore.from_existing_collection(
            embedding=_EMBEDDINGS,
            url=_qdrant_url(),
            api_key=_qdrant_api_key(),
            collection_name=collection_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name}' not found yet. Upload a document first.",
        ) from e


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>RAG Demo</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; max-width: 900px; margin: 40px auto; padding: 0 16px; }
      input, button, textarea { font: inherit; }
      .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
      .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin: 16px 0; }
      .muted { color: #6b7280; }
      pre { white-space: pre-wrap; word-break: break-word; background: #0b1020; color: #e5e7eb; padding: 12px; border-radius: 10px; }
      label { display: block; margin-bottom: 6px; }
      textarea { width: 100%; min-height: 96px; }
    </style>
  </head>
  <body>
    <h1>RAG Demo</h1>
    <p class="muted">Upload a PDF/TXT, then ask a question.</p>

    <div class="card">
      <h2>Upload</h2>
      <div class="row">
        <input id="file" type="file" />
        <input id="collection" type="text" value="learning-rag" placeholder="collection name" />
        <button id="uploadBtn">Upload & Index</button>
      </div>
      <p class="muted">Chunk size: <input id="chunkSize" type="number" value="1000" style="width: 100px;" />
      Overlap: <input id="chunkOverlap" type="number" value="400" style="width: 100px;" /></p>
      <pre id="uploadOut"></pre>
    </div>

    <div class="card">
      <h2>Ask</h2>
      <label for="question">Question</label>
      <textarea id="question" placeholder="Ask something about your uploaded doc..."></textarea>
      <div class="row">
        <button id="askBtn">Ask</button>
        <span class="muted">Top K: <input id="topK" type="number" value="4" style="width: 80px;" /></span>
      </div>
      <pre id="askOut"></pre>
    </div>

    <script>
      const uploadBtn = document.getElementById('uploadBtn');
      const askBtn = document.getElementById('askBtn');
      const uploadOut = document.getElementById('uploadOut');
      const askOut = document.getElementById('askOut');
      let activeDocId = null;

      uploadBtn.onclick = async () => {
        uploadOut.textContent = 'Uploading...';
        const file = document.getElementById('file').files[0];
        const collection = document.getElementById('collection').value || 'learning-rag';
        const chunkSize = document.getElementById('chunkSize').value || '1000';
        const chunkOverlap = document.getElementById('chunkOverlap').value || '400';
        if (!file) { uploadOut.textContent = 'Choose a file first.'; return; }

        const fd = new FormData();
        fd.append('file', file);
        fd.append('collection_name', collection);
        fd.append('chunk_size', chunkSize);
        fd.append('chunk_overlap', chunkOverlap);

        const res = await fetch('/upload', { method: 'POST', body: fd });
        const data = await res.json();
        uploadOut.textContent = JSON.stringify(data, null, 2);
        if (res.ok && data.doc_id) {
          activeDocId = data.doc_id;
          localStorage.setItem(`rag_doc_id_${collection}`, activeDocId);
        }
      };

      askBtn.onclick = async () => {
        askOut.textContent = 'Thinking...';
        const question = document.getElementById('question').value;
        const collection = document.getElementById('collection').value || 'learning-rag';
        const topK = Number(document.getElementById('topK').value || '4');
        if (!question.trim()) { askOut.textContent = 'Type a question first.'; return; }
        if (!activeDocId) {
          activeDocId = localStorage.getItem(`rag_doc_id_${collection}`);
        }

        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, collection_name: collection, top_k: topK, doc_id: activeDocId })
        });
        const data = await res.json();
        if (!res.ok) {
          askOut.textContent = JSON.stringify(data, null, 2);
          return;
        }
        const sources = (data.sources || []).map(s => `- page: ${s.page_label} | file: ${s.source}`).join('\\n');
        askOut.textContent = `Answer:\\n${data.answer || ''}\\n\\nSources:\\n${sources || '(none)'}`;
      };
    </script>
  </body>
</html>
"""


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    collection_name: str = Form("learning-rag"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(400),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md are supported")

    uploads_dir = Path(__file__).parent / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = uploads_dir / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    doc_id = uuid.uuid4().hex
    try:
        result = index_file(
            saved_path,
            collection_name=collection_name,
            doc_id=doc_id,
            original_filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return {
            "ok": True,
            "collection_name": result["collection_name"],
            "doc_id": doc_id,
            "file": result["file"],
            "original_filename": file.filename,
            "pages": result["pages"],
            "chunks": result["chunks"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class AskRequest(BaseModel):
    question: str
    collection_name: str = "learning-rag"
    top_k: int = 4
    doc_id: str | None = None


@app.post("/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    if not payload.doc_id:
        raise HTTPException(status_code=400, detail="Upload a document first (doc_id is missing).")

    vector_db = _get_vector_store(payload.collection_name)
    _ensure_doc_id_payload_index(payload.collection_name)
    qdrant_filter = None
    qdrant_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.doc_id",
                match=qdrant_models.MatchValue(value=payload.doc_id),
            )
        ]
    )
    search_result = vector_db.similarity_search(
        query=question, k=payload.top_k, filter=qdrant_filter
    )
    if not search_result:
        return {"answer": "I don't know based on the document.", "sources": []}

    context_parts = []
    sources = []
    seen_sources = set()
    for item in search_result:
        page_label = None
        source = None
        if isinstance(getattr(item, "metadata", None), dict):
            page_label = item.metadata.get("page_label") or item.metadata.get("page")
            source = item.metadata.get("source")
        context_parts.append(
            f"Page Number: {page_label}\nSource: {source}\nContent:\n{item.page_content}"
        )
        key = (str(page_label), str(source))
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"page_label": page_label, "source": source})

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = f"""You are a helpful assistant. Answer the user using only the context.
Do not add facts, examples, or explanations that are not explicitly present in the context.
If the answer is not in the context, say you don't know based on the document.
When you make a claim, include the page number(s) in parentheses like (Page 48).

Context:
{context}
"""

    response = _OPENAI.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "nodejs.pdf"
    result = index_file(pdf_path, collection_name="learning-rag")
    print(result["pages"])
    print("Indexing documents done.")
