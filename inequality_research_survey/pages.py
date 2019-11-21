from otree.api import Currency as c, currency_range
from ._builtin import Page, WaitPage
from .models import Constants

import yaml

class WelcomePage(Page):
    pass

class DemographicQuestions(Page):
    form_model = 'player'
    form_fields = ['gender', 'age', 'nationality', 'education', 'field_of_work', 'political_party']

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
    form_model = 'player'
    form_fields = [
        'undefined_question_1',
        'undefined_question_2',
        'undefined_question_3',
        ]

    def vars_for_template(self):
        stream = open("fixtures/questions.yaml", 'r')
        questions = yaml.safe_load(stream)
        if self.player.treatment == "treatment_1" or self.player.treatment == "treatment_2":
            statement_a = questions.get("vignette").get("a").get("statement")
            random_question = questions.get("vignette").get("a").get("random")
            question_1 = questions.get("vignette").get("a").get("case_questions")[0]
            question_2 = questions.get("vignette").get("a").get("case_questions")[1]
            question_3 = questions.get("vignette").get("a").get("case_questions")[2]
            max = 800
        elif self.player.treatment == "treatment_3" or self.player.treatment == "treatment_4":
            statement_a = questions.get("vignette").get("b").get("statement")
            random_question = questions.get("vignette").get("b").get("random")
            question_1 = questions.get("vignette").get("b").get("case_questions")[0]
            question_2 = questions.get("vignette").get("b").get("case_questions")[1]
            question_3 = questions.get("vignette").get("b").get("case_questions")[2]
            max = 50
        else:
            statement_a = questions.get("vignette").get("c").get("statement")
            random_question = questions.get("vignette").get("c").get("random")
            question_1 = questions.get("vignette").get("c").get("case_questions")[0]
            question_2 = questions.get("vignette").get("c").get("case_questions")[1]
            question_3 = questions.get("vignette").get("c").get("case_questions")[2]
            max = 20
        return dict(
            statement_a=statement_a,
            random_question=random_question,
            question_1 = question_1,
            question_2 = question_2,
            question_3 = question_3,
            top = max,
            )

class ExitPage(Page):
    pass

page_sequence = [
    WelcomePage,
    DemographicQuestions,
    Beliefs,
    Vignette_1,
    ExitPage,
    ]
