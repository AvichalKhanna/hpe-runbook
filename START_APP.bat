@echo off
setlocal enabledelayedexpansion
title Stratum AI - Startup

:: 1. Fix working directory
cd /d "%~dp0"

echo ========================================================
echo   Stratum AI - Starting Runbook RAG Chatbot
echo ========================================================
echo.

:: 2. Check and activate venv
if exist "venv\Scripts\activate.bat" goto :VENV_OK
echo [ERROR] Virtual environment not found.
echo Please run INSTALL.bat first.
goto :FAIL
:VENV_OK
:: call venv\Scripts\activate.bat
:: NOT USING THE VENV TO AVOID ISSUE WITH VULKAN BUILD

:: 3. Verify required models
if exist "models\MiniLM-L6-v2\model.safetensors" goto :EMBED_OK
if exist "models\MiniLM-L6-v2\pytorch_model.bin" goto :EMBED_OK
echo [ERROR] Embedding model missing. Run INSTALL.bat.
goto :FAIL
:EMBED_OK

if exist "models\bge-reranker-base\model.safetensors" goto :RERANK_OK
if exist "models\bge-reranker-base\pytorch_model.bin" goto :RERANK_OK
echo [WARNING] Reranker model missing. System will run in degraded mode.
:RERANK_OK

if exist "models\Qwen2.5-3B-Instruct-Q3_K_M.gguf" goto :LLM_OK
echo [ERROR] LLM model missing. Run INSTALL.bat.
goto :FAIL
:LLM_OK

:: 4. Verify indexes
if exist "index_store\faiss.index" goto :INDEX_OK
echo [INFO] Index not found. Building full index now first time...
python ingest.py
if %errorlevel% neq 0 goto :INGEST_FAIL
goto :INDEX_OK

:INGEST_FAIL
echo [ERROR] Ingestion failed.
goto :FAIL

:INDEX_OK
:: 5. Warm up models
echo [INFO] Warming up embedding model...
python -c "from pipeline.embedder import get_embedder; get_embedder().encode(['warmup'])" >nul 2>&1

:: 6. Print startup summary
echo.
echo ┌─────────────────────────────────────┐
echo │   STRATUM AI — SYSTEM READY         │
echo │   Backend:    http://127.0.0.1:8000 │
echo │   Embedding:  MiniLM-L6-v2          │
echo │   Reranker:   bge-reranker-base     │
echo │   LLM:        Qwen2.5-3B GGUF       │
echo └─────────────────────────────────────┘
echo.
echo [INFO] Starting Uvicorn server...
echo.

:: 7. Start server
python -m uvicorn server:app --port 8000

:: If uvicorn exits, pause so the user can read the error
echo.
echo [INFO] Server stopped.
pause
exit /b 0

:FAIL
echo.
echo [ERROR] Startup failed. Please check the logs above.
pause
exit /b 1
