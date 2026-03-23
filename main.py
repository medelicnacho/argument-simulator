# ============================================================
# main.py  —  Entry point for the social simulation.
#
# UPDATED:
#   Creates one shared conversation_history list and passes it
#   to both agents so they both read and write the same history.
# ============================================================

from multiprocessing.managers import BaseManager

from panda3d.core import AmbientLight, Vec4, Vec3, LineSegs
from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from npclogic import Agent


class QueueManager(BaseManager):
    pass

QueueManager.register("get_request_queue")
QueueManager.register("get_response_queue")


def connect_to_brain_server():
    print("Connecting to brain_server.py on port 50000...")
    manager = QueueManager(address=("127.0.0.1", 50000), authkey=b"consciousness")
    manager.connect()
    print("Connected to brain_server.py\n")
    return manager.get_request_queue(), manager.get_response_queue()


def make_synapse(app, pos_a, pos_b):
    ls = LineSegs()
    ls.setThickness(1.5)
    ls.setColor(Vec4(0.25, 0.25, 0.45, 0.7))
    ls.moveTo(pos_a)
    ls.drawTo(pos_b)
    app.render.attachNewNode(ls.create())


class Game(ShowBase):

    def __init__(self, request_queue, response_queue):
        ShowBase.__init__(self)
        self.setBackgroundColor(0.02, 0.02, 0.06, 1)

        self.turn = "vera"

        self.request_queue  = request_queue
        self.response_queue = response_queue

        # ── shared conversation history ───────────────────────────────────────────────
        # One list passed to BOTH agents. Both agents append their sentences here.
        # This means phi-3 always sees the full back-and-forth argument.
        self.conversation_history = []                                  # starts empty — fills up as they talk

        self._setup_lighting()
        self._setup_camera()
        self._create_agents()
        self._connect_agents()
        self._draw_synapse()

        print("=" * 55)
        print(" Social sim running. Watch the terminal.")
        print("=" * 55 + "\n")

        self.taskMgr.doMethodLater(3.0, self._kick_off, "kick_off")
        self.taskMgr.add(self.update, "update")


    def _kick_off(self, task):
        self.vera._request_speech(0)
        return Task.done


    def _setup_lighting(self):
        alight = AmbientLight("ambient")
        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
        self.render.setLight(self.render.attachNewNode(alight))


    def _setup_camera(self):
        self.camera.setPos(0, -11, 3.5)
        self.camera.lookAt(0, 0, 0.5)


    def _create_agents(self):

        self.vera = Agent(
            app                     = self,
            name                    = "vera",
            position                = Vec3(-2.8, 0, 0),
            colour                  = Vec4(0.3, 0.6, 1.0, 1),
            thought_delay           = 20.0,
            personality_bias        = {"strength": 1.5},
            personality_description = (
                "you are animal rights activist. "
                "you complain alot. "
                "your emotional unstable. "
                "your a self richious vegan."
                "you hate meat eaters"
            ),
            request_queue           = self.request_queue,
            response_queue          = self.response_queue,
            voice                   = "en-US-AriaNeural",               # sharp female voice for Vera
            conversation_history    = self.conversation_history,        # shared history — same list as Echo
        )

        self.echo = Agent(
            app                     = self,
            name                    = "echo",
            position                = Vec3(2.8, 0, 0),
            colour                  = Vec4(1.0, 0.55, 0.15, 1),
            thought_delay           = 20.0,
            personality_bias        = {"strength": 1.5},
            personality_description = (
                "you are redneck truck driver. "
                "you are proud of your pet cock richard. "
                "you love cock fighitng. "
                "veganism annoys you."
                "you love texus bbq"
            ),
            request_queue           = self.request_queue,
            response_queue          = self.response_queue,
            voice                   = "en-US-GuyNeural",                # deeper male voice for Echo
            conversation_history    = self.conversation_history,        # same shared list as Vera
        )


    def _connect_agents(self):
        self.vera.set_listener(self.echo)
        self.echo.set_listener(self.vera)


    def _draw_synapse(self):
        make_synapse(self, Vec3(-2.8, 0, 0), Vec3(2.8, 0, 0))


    def update(self, task):
        dt = globalClock.getDt()
        self.vera.update(dt)
        self.echo.update(dt)
        return Task.cont


if __name__ == "__main__":
    request_queue, response_queue = connect_to_brain_server()
    Game(request_queue, response_queue).run()
