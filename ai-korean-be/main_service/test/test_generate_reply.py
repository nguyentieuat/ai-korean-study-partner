from flask import Flask, request, jsonify
from flask_cors import CORS
from grammar_checker.grammar_checker import check_grammar
from dialogue_engine.dialog_generator import generate_reply

prompt_history = [
    {"role": "system", "content": "You are a friendly Korean speaking partner."},
    {"role": "user", "content": "안녕하세요."}
]

reply = generate_reply(prompt_history)
print("Bot:", reply)