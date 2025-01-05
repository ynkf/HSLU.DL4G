import copy
import numpy as np
from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
from jass.game.game_sim import GameSim
import logging
from tensorflow.keras.models import load_model
import time

# Configure logging
def configure_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger

logger = configure_logging()

# Card score definitions
CARD_SCORES = {
    "trump": [15, 10, 7, 25, 6, 19, 5, 5, 5],
    "no_trump": [9, 7, 5, 2, 1, 0, 0, 0, 0],
    "obenabe": [14, 10, 8, 7, 5, 0, 5, 0, 0],
    "uneufe": [0, 2, 1, 1, 5, 5, 7, 9, 11],
}

def calculate_card_score(card, trump_suit, is_trump):
    card_offset = offset_of_card[card]
    score_type = "trump" if is_trump else "no_trump"
    return CARD_SCORES[score_type][card_offset]

def highest_card_in_trick(trick, obs):
    """Determine the highest card and the winner of the trick."""
    trump = obs.trump
    color_of_first_card = color_of_card[trick[0]]
    highest_card, winner = trick[0], 0
    trump_played, trump_card = False, None

    for i, card in enumerate(trick):
        if card == -1:
            continue
        if color_of_card[card] == trump:
            if trump_played and lower_trump[card, trump_card]:
                trump_card, winner = card, i
            elif not trump_played:
                trump_card, trump_played, winner = card, True, i
        elif not trump_played and color_of_card[card] == color_of_first_card and card < highest_card:
            highest_card, winner = card, i

    return trump_card if trump_played else highest_card, winner


class AgentMCTSTrumpSchieber(Agent):
    def __init__(self, n_simulations=1, n_determinizations=90):
        super().__init__()
        self._rule = RuleSchieber()
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
        self.model = load_model('trump_model.h5')

    def action_play_card(self, obs):
        """Choose the best card to play."""
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]

        trump_suit = obs.trump
        current_points = sum(calculate_card_score(card, trump_suit, color_of_card[card] == trump_suit)
                             for card in obs.current_trick if card != -1)

        for card in valid_card_indices:
            card_color = color_of_card[card]
            if current_points >= 15 and card_color == trump_suit:
                trick = self._simulate_trick(obs, card)
                if highest_card_in_trick(trick, obs)[0] == card:
                    return card

        card_scores = self._simulate_with_mcts(obs, valid_card_indices)
        return valid_card_indices[np.argmax(card_scores)]

    def _simulate_trick(self, obs, card):
        trick = copy.deepcopy(obs.current_trick)
        trick[len([c for c in trick if c != -1])] = card
        return trick

    def _simulate_with_mcts(self, obs, valid_card_indices):
        card_scores = np.zeros(len(valid_card_indices))
        start_time = time.time()
        while time.time() - start_time < 9.5:
            determinized_hands = self._create_determinization(obs)
            card_scores += self._evaluate_mcts(determinized_hands, obs, valid_card_indices)
        return card_scores

    def _create_determinization(self, obs):
        return self._deal_unplayed_cards(obs)

    def _deal_unplayed_cards(self, obs):
        played_cards = {card for round_cards in obs.tricks for card in round_cards if card != -1}
        unplayed_cards = list(set(range(36)) - played_cards - set(convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)))
        np.random.shuffle(unplayed_cards)

        hands = np.zeros((4, 36), dtype=int)
        hands[obs.player] = obs.hand
        first_player = obs.trick_first_player[obs.nr_tricks]
        num_cards_per_player = 9 - obs.nr_tricks

        for i in range(4):
            if i != obs.player:
                player_cards = unplayed_cards[:num_cards_per_player]
                hands[i, player_cards] = 1
                unplayed_cards = unplayed_cards[num_cards_per_player:]

        return hands

    def _evaluate_mcts(self, hands, obs, valid_card_indices):
        scores = np.zeros(len(valid_card_indices))
        for _ in range(self.n_simulations):
            for i, card in enumerate(valid_card_indices):
                sim_game = GameSim(rule=self._rule)
                sim_game.init_from_state(copy.deepcopy(obs))
                sim_game._state.hands = copy.deepcopy(hands)
                sim_game.action_play_card(card)
                while not sim_game.is_done():
                    valid_cards_sim = self._rule.get_valid_cards_from_obs(sim_game.get_observation())
                    sim_game.action_play_card(np.random.choice(np.flatnonzero(valid_cards_sim)))
                scores[i] += sim_game.state.points[self._team(obs.player)]
        return scores

    def _team(self, player):
        return player % 2

    def action_trump(self, obs):
        """Decide on the trump suit."""
        hand = np.append(obs.hand, 1 if obs.forehand == -1 else 0).reshape(1, -1)
        probabilities = self.model.predict(hand)[0]
        trump_push_index = probabilities.argmax()

        if probabilities[trump_push_index] > 0.8:
            return trump_push_index

        ignored_indices = ["trump_OBE_ABE", "trump_UNE_UFE"]
        scores = [p if i not in ignored_indices else -1 for i, p in enumerate(probabilities)]
        return scores.index(max(scores))
