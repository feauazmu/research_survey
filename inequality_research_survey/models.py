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

def seven_options_likert_scale(question):
    # Define a Likert-type scale with 7 seven options
    return models.IntegerField(
        choices = [
            [7, "Strongly agree"],
            [6, "Agree"],
            [5, "Somewhat agree"],
            [4, "Neither agree nor disagree"],
            [3, "Somewhat disagree"],
            [2, "Disagree"],
            [1, "Strongly disagree"],
        ],
        label = question,
        widget = widgets.RadioSelect,
    )

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
        label = "Geschlecht:",
        choices = [
            [1, 'männlich'],
            [2, 'weiblich'],
            [3, 'divers'],
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )

    age = models.IntegerField(
        label = "Alter:",
        min = 18,
        max = 130,
        blank = False,
    )

    nationality = models.IntegerField(
        label = "Nationalität:",
        choices = [
            [1, 'deutsch'],
            [2, 'EU-Bürger'],
            [3, 'nicht EU-Bürger'],
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )

    education = models.IntegerField(
        label = "Was ist ihr höchster Bildungsabschluss:",
        choices = [
            [1, 'Hochschulabschluss'],
            [2, 'Berufsausbildung'],
            [3, 'Abitur'],
            [4, 'Realschulabschluss'],
            [5, 'Keiner']
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )


    """ Items of the Belief in Zero-Sum Game (BZSG) scale questionnaire.
    Taken from: Różycka-Tran, J., Boski, P., & Wojciszke, B. (2015).
    Belief in a zero-sum game as a social axiom: A 37-nation study. Journal of
    Cross-Cultural Psychology, 46(4), 525-548.

    """

    bzsg_1 = seven_options_likert_scale("Erfolge einiger Menschen sind normalerweise das Versagen anderer Menschen.")
    bzsg_2 = seven_options_likert_scale("Wird jemand reich, heißt es dass eine andere Person ärmer wird.")
    bzsg_3 = seven_options_likert_scale("Das Leben ist so gestaltet, dass wenn jemand gewinnt andere verlieren müssen.")
    bzsg_4 = seven_options_likert_scale("In den meisten Situationen sind die Interessen unterschiedlicher Menschen nicht zu vereinbaren.")
    bzsg_5 = seven_options_likert_scale("Das Leben ist wie ein Tennisspiel - man gewinnt nur wenn andere verlieren.")
    bzsg_6 = seven_options_likert_scale("Wenn einige Menschen ärmer werden bedeutet dies, dass andere Menschen reicher werden.")
    bzsg_7 = seven_options_likert_scale("Wenn ein Mensch viel für andere tut verliert er.")
    bzsg_8 = seven_options_likert_scale("Der Wohlstand weniger wird auf Kosten vieler erlangt.")

    # Perceptions on redistribution questions

    redistribution_1 = seven_options_likert_scale("Der Staat trägt die Verantwortug die Unterschiede zwischen Menschen mit hohem und Menschen mit niedrigerem Einkommen zu verringern.")
    redistribution_2 = seven_options_likert_scale("Der Staat sollte jedem der arbeiten möchte einen Arbeitsplatz bieten.")
    redistribution_3 = seven_options_likert_scale("Der Staat sollte jeden mit einem garantierten Grundeinkommen versorgen.")
    redistribution_4 = seven_options_likert_scale("Wir brauchen größere Einkommensunterschiede um Anreize für individuelles Bemühen zu schaffen.")
