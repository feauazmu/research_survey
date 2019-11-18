from otree.api import Currency as c, currency_range
from ._builtin import Page, WaitPage
from .models import Constants

import yaml

class DemographicQuestions(Page):
    form_model = 'player'
    form_fields = ['gender', 'age', 'nationality', 'education', 'field_of_work']

class Beliefs(Page):

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

class Vignette_1(Page):


    def vars_for_template(self):
        stream = open("fixtures/questions.yaml", 'r')
        questions = yaml.safe_load(stream)
        if self.player.treatment == "treatment_1" or self.player.treatment == "treatment_2":
            statement_a = questions.get("vignette").get("a").get("statement")
        elif self.player.treatment == "treatment_3" or self.player.treatment == "treatment_4":
            statement_a = questions.get("vignette").get("b").get("statement")
        else:
            statement_a = questions.get("vignette").get("c").get("statement")
        return dict(statement_a = statement_a)

page_sequence = [
    DemographicQuestions,
    Beliefs,
    Vignette_1,
    ]
