from otree.api import Currency as c, currency_range
from ._builtin import Page, WaitPage
from .models import Constants

import yaml

class WelcomePage(Page):
    pass

class DemographicQuestions(Page):
    form_model = 'player'
    form_fields = ['gender', 'age', 'nationality', 'education', 'field_of_work', 'political_party']

class Beliefs_1(Page):
    def is_displayed(self):
        return self.player.treatment == "treatment_1" or self.player.treatment == "treatment_3" or self.player.treatment == "treatment_5"

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

class Beliefs_2(Page):
    def is_displayed(self):
        return self.player.treatment == "treatment_2" or self.player.treatment == "treatment_4" or self.player.treatment == "treatment_6"

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
        'undefined_question_4',
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
            question_4 = questions.get("vignette").get("a").get("case_questions")[3]
            name_1 = questions.get("vignette").get("a").get("names")[0]
            name_2 = questions.get("vignette").get("a").get("names")[1]
            max = 800
        elif self.player.treatment == "treatment_3" or self.player.treatment == "treatment_4":
            statement_a = questions.get("vignette").get("b").get("statement")
            random_question = questions.get("vignette").get("b").get("random")
            question_1 = questions.get("vignette").get("b").get("case_questions")[0]
            question_2 = questions.get("vignette").get("b").get("case_questions")[1]
            question_3 = questions.get("vignette").get("b").get("case_questions")[2]
            question_4 = questions.get("vignette").get("b").get("case_questions")[3]
            name_1 = questions.get("vignette").get("b").get("names")[0]
            name_2 = questions.get("vignette").get("b").get("names")[1]
            max = 50
        else:
            statement_a = questions.get("vignette").get("c").get("statement")
            random_question = questions.get("vignette").get("c").get("random")
            question_1 = questions.get("vignette").get("c").get("case_questions")[0]
            question_2 = questions.get("vignette").get("c").get("case_questions")[1]
            question_3 = questions.get("vignette").get("c").get("case_questions")[2]
            question_4 = questions.get("vignette").get("c").get("case_questions")[3]
            name_1 = questions.get("vignette").get("c").get("names")[0]
            name_2 = questions.get("vignette").get("c").get("names")[1]
            max = 20
        return dict(
            statement_a=statement_a,
            random_question=random_question,
            question_1 = question_1,
            question_2 = question_2,
            question_3 = question_3,
            question_4 = question_4,
            name_1 = name_1,
            name_2 = name_2,
            top = max,
            )

class Vignette_2(Page):
    form_model = 'player'
    form_fields = [
        'zsg_question_1',
        'zsg_question_2',
        'zsg_question_3',
        'zsg_question_4',
        ]

    def vars_for_template(self):
        stream = open("fixtures/questions.yaml", 'r')
        questions = yaml.safe_load(stream)
        if self.player.treatment == "treatment_5" or self.player.treatment == "treatment_6":
            statement_a = questions.get("vignette").get("a").get("statement")
            random_question = questions.get("vignette").get("a").get("zsg")
            question_1 = questions.get("vignette").get("a").get("case_questions")[0]
            question_2 = questions.get("vignette").get("a").get("case_questions")[1]
            question_3 = questions.get("vignette").get("a").get("case_questions")[2]
            question_4 = questions.get("vignette").get("a").get("case_questions")[3]
            max = 800
        elif self.player.treatment == "treatment_1" or self.player.treatment == "treatment_2":
            statement_a = questions.get("vignette").get("b").get("statement")
            random_question = questions.get("vignette").get("b").get("zsg")
            question_1 = questions.get("vignette").get("b").get("case_questions")[0]
            question_2 = questions.get("vignette").get("b").get("case_questions")[1]
            question_3 = questions.get("vignette").get("b").get("case_questions")[2]
            question_4 = questions.get("vignette").get("b").get("case_questions")[3]
            max = 50
        else:
            statement_a = questions.get("vignette").get("c").get("statement")
            random_question = questions.get("vignette").get("c").get("zsg")
            question_1 = questions.get("vignette").get("c").get("case_questions")[0]
            question_2 = questions.get("vignette").get("c").get("case_questions")[1]
            question_3 = questions.get("vignette").get("c").get("case_questions")[2]
            question_4 = questions.get("vignette").get("c").get("case_questions")[3]
            max = 20
        return dict(
            statement_a=statement_a,
            random_question=random_question,
            question_1 = question_1,
            question_2 = question_2,
            question_3 = question_3,
            question_4 = question_4,
            top = max,
            )
class Vignette_3(Page):
    form_model = 'player'
    form_fields = [
        'nzsg_question_1',
        'nzsg_question_2',
        'nzsg_question_3',
        'nzsg_question_4',
        ]

    def vars_for_template(self):
        stream = open("fixtures/questions.yaml", 'r')
        questions = yaml.safe_load(stream)
        if self.player.treatment == "treatment_3" or self.player.treatment == "treatment_4":
            statement_a = questions.get("vignette").get("a").get("statement")
            random_question = questions.get("vignette").get("a").get("nzsg")
            question_1 = questions.get("vignette").get("a").get("case_questions")[0]
            question_2 = questions.get("vignette").get("a").get("case_questions")[1]
            question_3 = questions.get("vignette").get("a").get("case_questions")[2]
            question_4 = questions.get("vignette").get("a").get("case_questions")[3]
            max = 800
        elif self.player.treatment == "treatment_5" or self.player.treatment == "treatment_6":
            statement_a = questions.get("vignette").get("b").get("statement")
            random_question = questions.get("vignette").get("b").get("nzsg")
            question_1 = questions.get("vignette").get("b").get("case_questions")[0]
            question_2 = questions.get("vignette").get("b").get("case_questions")[1]
            question_3 = questions.get("vignette").get("b").get("case_questions")[2]
            question_4 = questions.get("vignette").get("b").get("case_questions")[3]
            max = 50
        else:
            statement_a = questions.get("vignette").get("c").get("statement")
            random_question = questions.get("vignette").get("c").get("nzsg")
            question_1 = questions.get("vignette").get("c").get("case_questions")[0]
            question_2 = questions.get("vignette").get("c").get("case_questions")[1]
            question_3 = questions.get("vignette").get("c").get("case_questions")[2]
            question_4 = questions.get("vignette").get("c").get("case_questions")[3]
            max = 20
        return dict(
            statement_a=statement_a,
            random_question=random_question,
            question_1 = question_1,
            question_2 = question_2,
            question_3 = question_3,
            question_4 = question_4,
            top = max,
            )

class ExitPage(Page):
    pass

page_sequence = [
    WelcomePage,
    DemographicQuestions,
    Beliefs_1,
    Vignette_1,
    Vignette_2,
    Vignette_3,
    Beliefs_2,
    ExitPage,
    ]
