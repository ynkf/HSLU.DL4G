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

# Configure logging to output to Jupyter Notebook
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Check if there are handlers already and clear them to prevent duplicate logs
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
else:
    logger.handlers.clear()  # Clear existing handlers
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# score if the color is trump
trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
# score if the color is not trump
no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
# score if obenabe is selected (all colors)
obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0,]
# score if uneufe is selected (all colors)
uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]

def calculate_trump_selection_score(cards, trump: int) -> int:
    score = 0
    for card_index in cards:
        card_offset = offset_of_card[card_index]
        if color_of_card[card_index] == trump:
            score += trump_score[card_offset]
        else:
            score += no_trump_score[card_offset]
    return score

def calculate_point_value(card, trump_suit):
    """
    Calculate the point value of a card, considering if it is a trump card or not.
    """
    card_offset = offset_of_card[card]
    if color_of_card[card] == trump_suit:
        return trump_score[card_offset]
    else:
        return no_trump_score[card_offset]

def highest_card_in_trick(trick, obs: GameObservation):
    amount_played_cards = len([card for card in trick if card != -1])
    trump = obs.trump
    color_of_first_card = color_of_card[trick[0]]
    if color_of_first_card == trump:
        # trump mode and first card is trump: highest trump wins
        winner = 0
        highest_card = trick[0]
        for i in range(1, amount_played_cards):
            # lower_trump[i,j] checks if j is a lower trump than i
            if color_of_card[trick[i]] == trump and lower_trump[trick[i], highest_card]:
                highest_card = trick[i]
                winner = i

        return highest_card, winner


    else:
        # trump mode, but different color played on first move, so we have to check for higher cards until
        # a trump is played, and then for the highest trump
        winner = 0
        highest_card = trick[0]
        trump_played = False
        trump_card = None
        for i in range(1, amount_played_cards):
            if color_of_card[trick[i]] == trump:
                if trump_played:
                    # second trump, check if it is higher
                    if lower_trump[trick[i], trump_card]:
                        winner = i
                        trump_card = trick[i]
                else:
                    # first trump played
                    trump_played = True
                    trump_card = trick[i]
                    winner = i
            elif trump_played:
                # color played is not trump, but trump has been played, so ignore this card
                pass
            elif color_of_card[trick[i]] == color_of_first_card:
                # trump has not been played and this is the same color as the first card played
                # so check if it is higher
                if trick[i] < highest_card:
                    highest_card = trick[i]
                    winner = i

        return highest_card, winner


class AgentMCTSTrumpSchieber(Agent):
    def __init__(self, n_simulations=1, n_determinizations=90):
        super().__init__()
        self._rule = RuleSchieber()
        self.n_simulations = n_simulations
        self.n_determinizations = n_determinizations
        self.model = load_model('trump_model.h5')

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select the best card to play using rule-based logic only.
        """
        # play single valid card available
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]

        # check if staeche or not
        trump_suit = obs.trump
        current_trick_points = sum(calculate_point_value(card, trump_suit) for card in obs.current_trick if card != -1)

        for card in valid_card_indices:
            card_suit = color_of_card[card]

            # falls eigene karte besser als gespielte karten -> stechen
            print(f'trick: {obs.current_trick}, points: {current_trick_points}')
            print(f'card: {card}, card color: {card_suit}, trup: {trump_suit}')
            if current_trick_points >= 15 and card_suit == trump_suit:
                trick = copy.deepcopy(obs.current_trick)
                trick[len([card for card in trick if card != -1])] = card
                new_highest_card, new_winner = highest_card_in_trick(trick, obs)
                print(f'new highest card: {new_highest_card}')
                if new_highest_card == card:
                    print(f'stab with: {card}')
                    return card

        for card in valid_card_indices:
            card_suit = color_of_card[card]

            trick = copy.deepcopy(obs.current_trick)
            card_index = len([card for card in trick if card != -1])
            trick[card_index] = card
            new_highest_card, new_winner = highest_card_in_trick(trick, obs)
            if new_highest_card == card:
                return card

        # If no decision from heuristics, use MCTS to determine the best card
        card_scores = np.zeros(len(valid_card_indices))
        start_time = time.time()

        while time.time() < start_time + 9.5:
            determinization_hands = self._create_determinization(obs)
            determinization_scores = self._run_mcts_for_determinization(determinization_hands, obs, valid_card_indices)
            card_scores += determinization_scores
            #logger.debug("Determinization %d: Scores from MCTS simulation: %s", determinization_idx, determinization_scores)

        # Select the card with the best score from MCTS
        best_card_index = np.argmax(card_scores)
        best_card = valid_card_indices[best_card_index]
        #logger.debug("MCTS complete. Best card chosen: %d with score %f", best_card, card_scores[best_card_index])

        return best_card


    def _create_determinization(self, obs: GameObservation) -> np.ndarray:
        """
        Create a determinized version of the game state by assigning random plausible hands to opponents.
        """
        hands = self._deal_unplayed_cards(obs)

        return hands

    def _deal_unplayed_cards(self, obs: GameObservation):
        played_cards_per_round = obs.tricks
        played_cards = set([card for round in played_cards_per_round for card in round if card != -1])

        rounds_started_by = obs.trick_first_player
        num_cards_per_player = np.full(4, (9 - obs.nr_tricks))

        first_player = rounds_started_by[obs.nr_tricks]
        for i in range(4):
            player = (first_player + i) % 4
            if obs.current_trick[i] != -1:
                num_cards_per_player[player] -= 1

        all_cards = set(range(36))

        unplayed_cards = list(all_cards - played_cards)
        opponents_unplayed_cards = list(set(unplayed_cards) - set(convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)))

        np.random.shuffle(opponents_unplayed_cards)

        hands = np.zeros(shape=[4, 36], dtype=np.int32)

        hands[obs.player] = obs.hand

        for player in range(4):
            if player != obs.player:
                players_random_cards = opponents_unplayed_cards[:num_cards_per_player[player]]
                hands[player, players_random_cards] = 1
                opponents_unplayed_cards = opponents_unplayed_cards[num_cards_per_player[player]:]

        return hands


    def _run_mcts_for_determinization(self, hands: np.ndarray, obs: GameObservation, valid_card_indices: np.ndarray) -> np.ndarray:
        """
        Run multiple MCTS simulations for a given determinization and return scores for each valid card.
        """
        card_scores = np.zeros(len(valid_card_indices))

        for _ in range(self.n_simulations):
            # For each valid card, simulate the outcome by reinitializing the game simulation
            for i, card in enumerate(valid_card_indices):
                sim_game = GameSim(rule=self._rule)
                sim_game.init_from_state(copy.deepcopy(obs))
                sim_game._state.hands = copy.deepcopy(hands)

                # Simulate playing the card
                sim_game.action_play_card(card)

                # Play out the rest of the game randomly
                while not sim_game.is_done():
                    valid_cards_sim = self._rule.get_valid_cards_from_obs(sim_game.get_observation())

                    # Check if there are any valid cards left
                    if np.flatnonzero(valid_cards_sim).size == 0:
                        # No valid cards, break out of the loop or handle the situation
                        break

                    # Randomly play a valid card
                    sim_game.action_play_card(np.random.choice(np.flatnonzero(valid_cards_sim)))

                # Update score based on the points scored for the simulation
                points = sim_game.state.points[self._team(obs.player)]
                card_scores[i] += points

        return card_scores

    def _team(self, player: int) -> int:
        """
        Determine the team number for the given player.
        Players 0 and 2 are in team 0, and players 1 and 3 are in team 1.
        """
        return player % 2

    def action_trump(self, obs: GameObservation) -> int:
        hand = obs.hand

        if obs.forehand == -1:
            hand = np.append(hand, 1)
        else:
            hand = np.append(hand, 0)
        logger.debug("Hand: " + str(hand))

        hand = hand.reshape(1, -1)

        model = self.model
        probabilities = model.predict(hand)
        logger.debug("Model probabilities: " + str(probabilities))


        trump_categories = [
            "trump_DIAMONDS",
            "trump_HEARTS",
            "trump_SPADES",
            "trump_CLUBS",
            "trump_OBE_ABE",
            "trump_UNE_UFE",
            "trump_PUSH",
        ]

        scores = probabilities[0]

        trump_push_index = trump_categories.index("trump_PUSH")
        ignored_indices = [
            trump_categories.index("trump_OBE_ABE"),
            trump_categories.index("trump_UNE_UFE"),
        ]

        if scores[trump_push_index] > 0.8:
            logger.debug("Decision: Push (Threshold exceeded)")
            return trump_push_index

        filtered_scores = [
            score if idx not in ignored_indices else -1 for idx, score in enumerate(scores)
        ]

        highest_score_index = filtered_scores.index(max(filtered_scores))
        logger.debug(f"Filtered scores: {filtered_scores}")
        logger.debug(f"Highest score index: {highest_score_index}")

        return highest_score_index