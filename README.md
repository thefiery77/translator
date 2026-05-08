cd backend
uv sync   # oppure: pip install -e .

# 3. Copia e compila .env
cp .env.example .env
# → inserisci OPENAI_API_KEY

# 4. Avvia server
uvicorn app.main:app --reload --port 8000

# 5. Esegui test (no API key necessaria)
pytest tests/