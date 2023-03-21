# <img src='' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> AlpacaPP Persona
 
Give OpenVoiceOS some sass with [AlpacaCPP](https://github.com/ggerganov/llama.cpp)

## Examples 
* "What is best in life?"
* "Do you like dogs"
* "Does God exist?"


## Usage

Spoken answers api

```python
from ovos_solver_alpacacpp import AlpacaCPPSolver

ALPACA_MODEL_FILE = "/./models/ggml-alpaca-7b-q4.bin"

bot = AlpacaCPPSolver({"model": ALPACA_MODEL_FILE})

sentence = bot.spoken_answer("Qual é o teu animal favorito?", {"lang": "pt-pt"})
# Meus animais favoritos são cães, gatos e tartarugas!

for q in ["Does god exist?",
          "what is the speed of light?",
          "what is the meaning of life?",
          "What is your favorite color?",
          "What is best in life?"]:
    a = bot.get_spoken_answer(q)
    print(q, a)
```