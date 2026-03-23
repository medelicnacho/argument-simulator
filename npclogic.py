# ============================================================
# npclogic.py  —  Defines a single NPC agent in the simulation.
#
# UPDATED:
#   - agents now store a shared conversation history (list of sentences)
#   - that history is passed to brain_server so phi-3 sees the whole argument
#   - the list of recent thought indices is passed to NeuralBrain
#     so thought selection is shaped by the whole conversation pattern
# ============================================================

import random
import asyncio
import threading
import tempfile
import os

import edge_tts
import sounddevice as sd
import soundfile as sf

from panda3d.core import Vec3, Vec4, PointLight, TextNode

from brain import NeuralBrain


THOUGHT_BANK = [
    "animals are innocent",                                    # 0
    "is veganism ethically superior?",                         # 1
    "chickens are awesome",                                    # 2
    "i don't like when people disagree with me",               # 3
    "is there a difference between eating meat and not",       # 4
    "its so hot outside",                                      # 5
    "im annoyed people don't understand my side",              # 6
    "is cock fighting ethical?",                               # 7
    "bbq is a way of life",                                    # 8
    "do animals feel pain the same way we do?",                # 9
    "people who eat meat don't think about consequences",      # 10
    "richard is the best cock in the county",                  # 11
    "the smell of a smokehouse is pure heaven",                # 12
    "factory farming is a disgrace",                           # 13
    "texas does everything better including food",             # 14
    "if you love animals how can you eat them",                # 15
]

CONVERSATION_HISTORY_LENGTH = 6        # how many past sentences to remember and pass to phi-3
THOUGHT_HISTORY_LENGTH      = 5        # how many past thought indices to pass to the neural network


def speak_in_thread(text, voice):
    """
    Speak a sentence out loud using edge-tts in a background thread.
    The game loop never freezes because audio runs separately.
    """

    def _run():
        async def _generate():
            tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)  # temp file for the audio
            tmp.close()
            communicate = edge_tts.Communicate(text, voice)                 # create TTS request
            await communicate.save(tmp.name)                                 # generate and save audio
            return tmp.name

        path = asyncio.run(_generate())                                      # run async TTS generation
        data, samplerate = sf.read(path)                                     # read audio into array
        sd.play(data, samplerate)                                            # play through speakers
        sd.wait()                                                            # wait until finished
        os.remove(path)                                                      # clean up temp file

    thread = threading.Thread(target=_run, daemon=True)                      # background thread so game doesn't freeze
    thread.start()


class Agent:
    """
    One NPC agent — sphere, brain, voice, and now a shared conversation memory.

    CHANGE FROM BEFORE:
    Each agent holds a reference to a shared conversation_history list.
    Both agents append to the same list so it contains the full back-and-forth.
    This list is passed to phi-3 so it can see the whole argument so far.
    The list of recent thought indices is passed to NeuralBrain so thought
    selection is shaped by the pattern of the whole conversation.
    """

    def __init__(self, app, name, position, colour, thought_delay,
                 personality_bias, personality_description,
                 request_queue, response_queue, voice,
                 conversation_history):
        """
        Build this agent.

        New parameter:
        conversation_history : list
            A shared list that both agents append their sentences to.
            Pass the SAME list object to both agents in main.py so they
            share one history rather than each having their own.
        """

        self.app                     = app
        self.name                    = name
        self.colour                  = colour
        self.energy                  = 0.0
        self.thought_delay           = thought_delay
        self.thought_timer           = 0.0
        self.personality_description = personality_description
        self.request_queue           = request_queue
        self.response_queue          = response_queue
        self.listener                = None
        self._pending_stimulus       = None
        self._waiting_for_response   = False
        self.last_thought_index      = 0
        self.last_sentence           = ""                               # last sentence this agent said
        self.voice                   = voice                            # edge-tts voice name
        self.conversation_history    = conversation_history             # shared list — both agents write here

        # thought index history — used to pass a pattern to NeuralBrain
        # this is separate from conversation_history — it stores indices not sentences
        self.thought_index_history   = []                               # list of recent thought indices from both agents

        self.brain = NeuralBrain(
            number_of_possible_thoughts = len(THOUGHT_BANK),
            personality_bias            = personality_bias,
        )

        self.node = app.loader.loadModel("models/misc/sphere")
        self.node.reparentTo(app.render)
        self.node.setPos(position)
        self.node.setScale(0.5)
        self.node.setColor(colour)

        plight = PointLight(name + "_light")
        plight.setColor(colour)
        self.light_node = app.render.attachNewNode(plight)
        self.light_node.setPos(position)
        app.render.setLight(self.light_node)

        name_text = TextNode(name + "_name")
        name_text.setText(name.capitalize())
        name_text.setAlign(TextNode.ACenter)
        name_text.setTextColor(Vec4(1, 1, 1, 1))
        self.name_np = app.render.attachNewNode(name_text)
        self.name_np.setPos(position + Vec3(0, 0, 1.0))
        self.name_np.setScale(0.2)
        self.name_np.setBillboardPointEye()

        speech_text = TextNode(name + "_speech")
        speech_text.setText("")
        speech_text.setAlign(TextNode.ACenter)
        speech_text.setTextColor(Vec4(1, 1, 1, 1))
        speech_text.setWordwrap(14)
        self.speech_np = app.render.attachNewNode(speech_text)
        self.speech_np.setPos(position + Vec3(0, 0, 1.6))
        self.speech_np.setScale(0.3)
        self.speech_np.setBillboardPointEye()
        self.speech_node = speech_text

        self.speech_timer    = 0.0
        self.speech_duration = 18.0


    def set_listener(self, other_agent):
        # Purpose: connect to the other agent so turns and stimuli can be passed
        self.listener = other_agent


    def _get_recent_thought_history(self):
        """
        Return the last THOUGHT_HISTORY_LENGTH thought indices as a list.

        If the history is shorter than THOUGHT_HISTORY_LENGTH, return what we have.
        This is passed to NeuralBrain.pick_next_thought() so the whole recent
        conversation pattern shapes what thought gets picked next.
        """
        return self.thought_index_history[-THOUGHT_HISTORY_LENGTH:]     # slice the last N indices


    def _get_conversation_context(self):
        """
        Build a formatted string of the last few sentences for the phi-3 prompt.

        Takes the last CONVERSATION_HISTORY_LENGTH entries from conversation_history
        and formats them as a readable back-and-forth so phi-3 can see the argument.

        Returns
        -------
        str
            A formatted conversation block e.g.:
            "Vera: BBQ is cruel.
             Echo: Texas BBQ is life.
             Vera: Animals feel pain!"
        """
        recent = self.conversation_history[-CONVERSATION_HISTORY_LENGTH:]   # get last N entries
        return "\n".join(recent)                                             # join into one string with newlines


    def _request_speech(self, stimulus_index):
        """
        Send a speech request to brain_server.

        CHANGE FROM BEFORE:
        - passes thought_index_history list to NeuralBrain instead of single index
        - passes conversation context string to brain_server so phi-3 sees history
        - records the stimulus index in thought_index_history
        """

        if self._waiting_for_response:                                  # already waiting — skip
            return
        if self.app.turn != self.name:                                  # not our turn — skip
            return

        # add the incoming stimulus to the shared thought history before picking
        self.thought_index_history.append(stimulus_index)               # record what we just heard
        if len(self.thought_index_history) > THOUGHT_HISTORY_LENGTH * 2:  # keep history from growing forever
            self.thought_index_history.pop(0)                           # drop oldest

        # pass the whole recent thought history to the neural network
        thought_index              = self.brain.pick_next_thought(self._get_recent_thought_history())
        self.last_thought_index    = thought_index
        self._waiting_for_response = True

        self.request_queue.put({
            "agent_name"          : self.name,
            "personality"         : self.personality_description,
            "thought_seed"        : THOUGHT_BANK[thought_index],
            "thought_index"       : thought_index,
            "last_sentence"       : self.listener.last_sentence if self.listener else "",   # what they just said
            "conversation_context": self._get_conversation_context(),   # the full recent argument history
        })


    def _check_response_queue(self):
        """
        Called every frame — check if brain_server sent back a sentence.

        CHANGE FROM BEFORE:
        - appends the new sentence to conversation_history with speaker label
        - appends the thought index to thought_index_history
        """

        if not self._waiting_for_response:
            return

        try:
            response = self.response_queue.get_nowait()
        except Exception:
            return

        if response["agent_name"] != self.name:
            self.response_queue.put(response)
            return

        text          = response["text"]
        thought_index = response["thought_index"]

        self.last_sentence = text                                       # save last sentence for the other agent to react to

        # append to shared conversation history with speaker label
        # e.g. "Vera: Factory farming is a disgrace."
        self.conversation_history.append(f"{self.name.capitalize()}: {text}")   # add to shared history
        if len(self.conversation_history) > CONVERSATION_HISTORY_LENGTH * 2:    # keep it from growing forever
            self.conversation_history.pop(0)                                     # drop oldest entry

        # add our chosen thought index to the shared thought pattern history
        self.thought_index_history.append(thought_index)               # record what we just said
        if len(self.thought_index_history) > THOUGHT_HISTORY_LENGTH * 2:
            self.thought_index_history.pop(0)                           # drop oldest

        self.speech_node.setText(text)                                  # show in speech bubble
        self.speech_timer          = self.speech_duration               # reset display timer
        self.energy                = 1.0                                # flash glow
        self._waiting_for_response = False                              # ready for next turn

        print(f"\n[{self.name.capitalize()}]: {text}")                  # print to terminal

        speak_in_thread(text, self.voice)                               # speak out loud in background

        if self.listener:
            self.listener._pending_stimulus = thought_index            # tell the other agent what to respond to
            # share our thought history with the listener so their brain also sees the full pattern
            self.listener.thought_index_history = self.thought_index_history.copy()
            self.app.turn = self.listener.name                          # pass the turn


    def update(self, dt):
        # Purpose: called every frame — check queue, tick timers, animate glow

        self._check_response_queue()

        self.thought_timer += dt
        if self.thought_timer >= self.thought_delay:
            self.thought_timer = 0.0
            if self._pending_stimulus is not None and self.app.turn == self.name:
                stimulus               = self._pending_stimulus
                self._pending_stimulus = None
                self._request_speech(stimulus)

        if self.speech_timer > 0:
            self.speech_timer -= dt
            if self.speech_timer <= 0:
                self.speech_node.setText("")

        if self.energy > 0:
            self.energy -= dt * 0.8
            self.energy  = max(0.0, self.energy)

        glow = max(self.energy, 0.08)
        self.node.setColor(Vec4(
            self.colour[0] * glow,
            self.colour[1] * glow,
            self.colour[2] * glow,
            1.0,
        ))
