import numpy as np
from copy import deepcopy
from typing import List, Tuple

from jass.game.game_util import convert_one_hot_encoded_cards_to_int_encoded_list
from jass.game.game_observation import GameObservation
from jass.game.const import offset_of_card, color_of_card, lower_trump, PUSH
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent

# Score constants
TRUMP_SCORE = [15, 10, 7, 25, 6, 19, 5, 5, 5]
NO_TRUMP_SCORE = [9, 7, 5, 2, 1, 0, 0, 0, 0]

def calculate_point_value(card: int, trump_suit: int) -> int:
    """Calculate the point value of a card, considering trump suit."""
    card_offset = offset_of_card[card]
    if color_of_card[card] == trump_suit:
        return TRUMP_SCORE[card_offset]
    return NO_TRUMP_SCORE[card_offset]

def highest_card_in_trick(trick: List[int], obs: GameObservation) -> Tuple[int, int]:
    """Determine the highest card and winner index in a trick."""
    trump = obs.trump
    first_card_color = color_of_card[trick[0]]
    highest_card, winner = trick[0], 0
    
    for i, card in enumerate(trick[1:], start=1):
        if card == -1:
            continue
        if color_of_card[card] == trump:
            if color_of_card[highest_card] != trump or lower_trump[card, highest_card]:
                highest_card, winner = card, i
        elif color_of_card[card] == first_card_color and color_of_card[highest_card] != trump:
            if lower_trump[highest_card, card]:
                highest_card, winner = card, i

    return highest_card, winner

def calculate_trump_selection_score(cards: List[int], trump: int) -> int:
    """Calculate the score if the trump suit is selected."""
    return sum(
        TRUMP_SCORE[offset_of_card[card]] if color_of_card[card] == trump else NO_TRUMP_SCORE[offset_of_card[card]]
        for card in cards
    )

class AgentRuleBased(Agent):
    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()

    def action_play_card(self, obs: GameObservation) -> int:
        """Select the best card to play using rule-based logic only."""
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        if len(valid_card_indices) == 1:
            return valid_card_indices[0]

        trump_suit = obs.trump
        current_trick_points = sum(calculate_point_value(card, trump_suit) for card in obs.current_trick if card != -1)
        highest_card, winner = highest_card_in_trick(obs.current_trick, obs)
        
        for card in valid_card_indices:
            if self._should_stab(obs, card, current_trick_points, trump_suit):
                return card

        return self._play_highest_card(obs, valid_card_indices, highest_card)

    def _should_stab(self, obs: GameObservation, card: int, trick_points: int, trump_suit: int) -> bool:
        """Determine if a trump card should be played to win the trick."""
        if trick_points >= 15 and color_of_card[card] == trump_suit:
            temp_trick = deepcopy(obs.current_trick)
            temp_trick[len([c for c in temp_trick if c != -1])] = card
            new_highest_card, _ = highest_card_in_trick(temp_trick, obs)
            return new_highest_card == card
        return False

    def _play_highest_card(self, obs: GameObservation, valid_card_indices: List[int], highest_card: int) -> int:
        """Play the highest card based on game rules."""
        for card in valid_card_indices:
            temp_trick = deepcopy(obs.current_trick)
            temp_trick[len([c for c in temp_trick if c != -1])] = card
            new_highest_card, _ = highest_card_in_trick(temp_trick, obs)
            if new_highest_card == card:
                return card
        return np.random.choice(valid_card_indices)

    def action_trump(self, obs: GameObservation) -> int:
        """Determine trump action based on score evaluation."""
        card_list = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        scores = [calculate_trump_selection_score(card_list, trump) for trump in range(4)]
        best_trump = scores.index(max(scores))
        return best_trump if scores[best_trump] > 68 else (PUSH if obs.forehand == -1 else best_trump)
