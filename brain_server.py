# ============================================================
# brain_server.py  —  Runs phi-3 as a separate process.
#
# UPDATED:
#   The prompt now includes the full recent conversation history
#   so phi-3 can see what has already been said and respond to
#   the argument as a whole, not just the last sentence.
# ============================================================

import os
import threading
from multiprocessing.managers import BaseManager
import multiprocessing
from llama_cpp import Llama


MODEL_PATH = os.path.join("models", "Phi-3-mini-4k-instruct-q4.gguf")


def run_phi3_loop(request_queue, response_queue):
    """
    Load phi-3 once, then loop forever processing speech requests.

    CHANGE FROM BEFORE:
    The prompt now includes conversation_context — the last several
    sentences from both agents. phi-3 sees the whole argument so far
    and generates a reply that fits into that ongoing conversation.
    """

    print("[brain_server] Loading phi-3...")
    llm = Llama(
        model_path = MODEL_PATH,
        n_ctx      = 1024,                                  # increased from 512 — need more room for conversation history
        n_threads  = 4,
        verbose    = False,
    )
    print("[brain_server] phi-3 ready. Waiting for requests...\n")

    while True:
        request = request_queue.get()                       # block until a request arrives

        if request == "STOP":
            print("[brain_server] Shutting down.")
            break

        # ── build the conversation history block ──────────────────────────────────────
        context = request.get("conversation_context", "")              # get history string — empty if none yet

        if context:                                                     # if there is history...
            history_block = (
                f"Here is the conversation so far:\n"
                f"{context}\n\n"                                        # show the last N lines of the argument
            )
        else:                                                           # first turn — no history yet
            history_block = ""                                          # leave blank

        # ── build the full phi-3 prompt ───────────────────────────────────────────────
        prompt = (
            f"<|system|>\n"
            f"You are {request['agent_name']}. {request['personality']}\n"
            f"Respond with ONE short sentence (max 15 words). "
            f"No quotes. No explanation. Just the sentence.\n"
            f"<|end|>\n"
            f"<|user|>\n"
            f"{history_block}"                                          # conversation history goes here
            f"The other person just said: {request['last_sentence']}\n"
            f"Now express this thought in your voice and react to what they said: {request['thought_seed']}\n"
            f"<|end|>\n"
            f"<|assistant|>\n"
        )

        try:
            result = llm(
                prompt,
                max_tokens  = 40,
                temperature = 0.85,
                stop        = ["\n", "<|end|>"],
            )
            text = result["choices"][0]["text"].strip()
        except Exception:
            text = f"({request['thought_seed']})"

        response_queue.put({
            "agent_name"   : request["agent_name"],
            "text"         : text,
            "thought_index": request["thought_index"],
        })


request_queue  = multiprocessing.Queue()
response_queue = multiprocessing.Queue()


class QueueManager(BaseManager):
    pass

QueueManager.register("get_request_queue",  callable=lambda: request_queue)
QueueManager.register("get_response_queue", callable=lambda: response_queue)


if __name__ == "__main__":

    phi3_thread = threading.Thread(
        target = run_phi3_loop,
        args   = (request_queue, response_queue),
        daemon = True,
    )
    phi3_thread.start()

    manager = QueueManager(address=("127.0.0.1", 50000), authkey=b"consciousness")
    server  = manager.get_server()
    print("[brain_server] Queue manager listening on port 50000")
    server.serve_forever()
