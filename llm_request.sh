#!/bin/bash

# Usage: ./llm_request.sh <server_ip> "<prompt>" [max_tokens]

SERVER_IP="$1"
PROMPT="$2"
MAX_TOKENS="${3:-32}"  # Default to 32 if not provided

if [ -z "$SERVER_IP" ] || [ -z "$PROMPT" ]; then
  echo "Usage: $0 <server_ip> \"<prompt>\" [max_tokens]"
  exit 1
fi

RESPONSE=$(curl -s "http://${SERVER_IP}:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
        --arg model "meta-llama/Llama-3.2-1B-Instruct" \
        --arg prompt "$PROMPT" \
        --argjson max_tokens "$MAX_TOKENS" \
        --argjson temperature 0 \
        '{model: $model, prompt: $prompt, max_tokens: $max_tokens, temperature: $temperature}')")


GENERATED_TEXT=$(echo "$RESPONSE" | jq -r '.choices[0].text' | sed 's/^[ \t]*//;s/[ \t]*$//')

# Extract only the generated text
echo -e "\n📨 Prompt: $PROMPT"
echo -e "🧠 Completion:\n$GENERATED_TEXT\n"
