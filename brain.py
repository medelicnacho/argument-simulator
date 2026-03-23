# ============================================================
# brain.py  —  Neural network that picks which thought an agent has next.
#
# UPDATED: pick_next_thought() now takes a LIST of recent thought
# indices instead of a single index. The whole conversation pattern
# shapes the next thought, not just the last thing said.
# ============================================================

import numpy
import random


class NeuralBrain:
    """
    A two-layer neural network that picks the next thought.

    CHANGE FROM BEFORE:
    pick_next_thought() now accepts a list of recent thought indices.
    It blends them into one input vector so the network responds to
    the pattern of the whole recent conversation.
    """

    def __init__(self, number_of_possible_thoughts, personality_bias):

        self.number_of_possible_thoughts = number_of_possible_thoughts
        self.personality_bias            = personality_bias

        number_of_hidden_neurons = 24

        self.input_to_hidden_weights = numpy.random.randn(
            number_of_possible_thoughts,
            number_of_hidden_neurons
        ) * 0.4

        self.hidden_to_output_weights = numpy.random.randn(
            number_of_hidden_neurons,
            number_of_possible_thoughts
        ) * 0.4

        self.current_mood                  = random.uniform(0.4, 0.8)
        self.recent_thought_history        = []
        self.how_many_thoughts_to_remember = 3


    def pick_next_thought(self, thought_history):
        """
        Run a forward pass and return the next thought index.

        CHANGE FROM BEFORE:
        thought_history is now a LIST of recent thought indices
        e.g. [3, 7, 2, 11] instead of a single number.

        We add up the one-hot vectors for every index in the list
        and normalise — giving the network a blended signal of the
        whole recent conversation pattern.

        Parameters
        ----------
        thought_history : list of int
            Recent thought indices from the conversation. Most recent last.

        Returns
        -------
        int
            The index of the next thought to express.
        """

        # ── step 1: build blended input from conversation history ─────────────────────
        input_vector = numpy.zeros(self.number_of_possible_thoughts)    # all zeros to start
        for index in thought_history:                                    # for each thought in the history...
            input_vector[index] += 1.0                                  # add 1 to that thought's slot
        if input_vector.sum() > 0:                                      # avoid divide by zero
            input_vector /= input_vector.sum()                          # normalise so it sums to 1

        # ── step 2: input → hidden ────────────────────────────────────────────────────
        hidden_layer_raw = numpy.dot(input_vector, self.input_to_hidden_weights)

        # ── step 3: ReLU ─────────────────────────────────────────────────────────────
        hidden_layer_activated = numpy.maximum(0, hidden_layer_raw)     # clamp negatives to zero

        # ── step 4: personality boost ─────────────────────────────────────────────────
        personality_boost             = self._calculate_personality_boost(hidden_layer_activated)
        hidden_layer_with_personality = hidden_layer_activated + personality_boost

        # ── step 5: hidden → output ───────────────────────────────────────────────────
        output_scores_raw = numpy.dot(hidden_layer_with_personality, self.hidden_to_output_weights)

        # ── step 6: mood shift ────────────────────────────────────────────────────────
        output_scores_with_mood = output_scores_raw + self.current_mood

        # ── step 7: block recently said thoughts ──────────────────────────────────────
        for recently_said_index in self.recent_thought_history:
            output_scores_with_mood[recently_said_index] = -999.0       # make impossible to pick

        # ── step 8: softmax ───────────────────────────────────────────────────────────
        scores_shifted              = output_scores_with_mood - numpy.max(output_scores_with_mood)
        scores_exponentiated        = numpy.exp(scores_shifted)
        probability_of_each_thought = scores_exponentiated / scores_exponentiated.sum()

        # ── step 9: weighted random sample ───────────────────────────────────────────
        chosen_thought_index = numpy.random.choice(
            self.number_of_possible_thoughts,
            p=probability_of_each_thought
        )

        # ── step 10: update state ─────────────────────────────────────────────────────
        self._remember_thought(chosen_thought_index)
        self._drift_mood()

        return chosen_thought_index


    def _calculate_personality_boost(self, hidden_layer):
        total_boost_strength = sum(self.personality_bias.values())
        if total_boost_strength == 0:
            return numpy.zeros_like(hidden_layer)
        return hidden_layer * (total_boost_strength * 0.15)


    def _remember_thought(self, thought_index):
        self.recent_thought_history.append(thought_index)
        if len(self.recent_thought_history) > self.how_many_thoughts_to_remember:
            self.recent_thought_history.pop(0)


    def _drift_mood(self):
        self.current_mood += random.uniform(-0.08, 0.08)
        self.current_mood  = max(0.15, min(1.0, self.current_mood))
