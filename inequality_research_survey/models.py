from otree.api import (
    models,
    widgets,
    BaseConstants,
    BaseSubsession,
    BaseGroup,
    BasePlayer,
    Currency as c,
    currency_range,
)


author = 'Your name here'

doc = """
Your app description
"""


class Constants(BaseConstants):
    name_in_url = 'inequality_research_survey'
    players_per_group = None
    num_rounds = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Demographic questions.
    gender = models.IntegerField(
        label = "What is your gender?",
        choices = [
            [1, 'Male'],
            [2, 'Female'],
            [3, 'Other'],
        ],
        widget = widgets.RadioSelect,
        blank = True,
    )

    age = models.IntegerField(
        label = "What is your age?",
        min = 18,
        max = 130,
        blank = True,
    )
