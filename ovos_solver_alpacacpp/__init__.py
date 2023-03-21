from neon_solvers import AbstractSolver
from os.path import dirname
from ovos_solver_alpacacpp.personas import OVOSAlpaca


class AlpacaCPPSolver(AbstractSolver):
    def __init__(self, config=None):
        super().__init__(name="AlpacaCPP", priority=94, config=config,
                         enable_cache=False, enable_tx=True)
        checkpoint = self.config.get("model")
        self.model = OVOSAlpaca(checkpoint)

    # officially exported Solver methods
    def get_spoken_answer(self, query, context=None):
        return self.model.ask(query)


if __name__ == "__main__":
    ALPACA_MODEL_FILE = f"/{dirname(dirname(__file__))}/models/ggml-alpaca-7b-q4.bin"

    bot = AlpacaCPPSolver({"model": ALPACA_MODEL_FILE})

    sentence = bot.spoken_answer("Qual é o teu animal favorito?", {"lang": "pt-pt"})
    print(sentence)

    for q in ["Does god exist?",
              "what is the speed of light?",
              "what is the meaning of life?",
              "What is your favorite color?",
              "What is best in life?"]:
        a = bot.get_spoken_answer(q)
        print(q, a)
