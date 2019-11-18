from otree.api import Currency as c, currency_range
from ._builtin import Page, WaitPage
from .models import Constants

import yaml

class DemographicQuestions(Page):
    form_model = 'player'
    form_fields = ['gender', 'age', 'nationality', 'education']

class Beliefs(Page):
    stream = open("fixtures/questions.yaml", 'r')
    questions = yaml.safe_load(stream)

    statement_a = questions.get("vignette").get("a").get("statement")
    form_model = 'player'
    form_fields = [
        # Beliefs in zero-sum game:

        'bzsg_1',
        'bzsg_2',
        'bzsg_3',
        'bzsg_4',
        'bzsg_5',
        'bzsg_6',
        'bzsg_7',
        'bzsg_8',

        # Beliefs in redistribution:

        'redistribution_1',
        'redistribution_2',
        'redistribution_3',
        'redistribution_4',
    ]

page_sequence = [
    DemographicQuestions,
    Beliefs,
    ]
