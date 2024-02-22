@echo off
curl -X POST -H "Content-Type: application/json" -d "{\"query\": \"%~1\"}" http://127.0.0.1:5000/query