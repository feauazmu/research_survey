from otree.api import Currency as c, currency_range
from ._builtin import Page, WaitPage
from .models import Constants


class DemographicQuestions(Page):
    form_model = 'player'
    form_fields = ['gender', 'age']

page_sequence = [DemographicQuestions]
