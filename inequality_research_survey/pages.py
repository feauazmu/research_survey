from otree.api import Currency as c, currency_range
from ._builtin import Page, WaitPage
from .models import Constants


class DemographicQuestions(Page):
    form_model = 'player'
    form_fields = ['gender', 'age', 'nationality', 'education']

class Beliefs(Page):
    form_model = 'player'
    form_fields = [
        'bzsg_1',
        'bzsg_2',
        'bzsg_3',
        'bzsg_4',
        'bzsg_5',
        'bzsg_6',
        'bzsg_7',
        'bzsg_8'
    ]

page_sequence = [
    DemographicQuestions,
    Beliefs,
    ]
