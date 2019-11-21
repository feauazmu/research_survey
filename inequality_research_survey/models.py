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
import yaml


author = 'Your name here'

doc = """
Your app description
"""

stream = open("fixtures/questions.yaml", 'r')
questions = yaml.safe_load(stream)

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

    def creating_session(self):
        import itertools
        treatments = itertools.cycle([
            'treatment_1',
            'treatment_2',
            'treatment_3',
            'treatment_4',
            'treatment_5',
            'treatment_6',
            ])
        for p in self.get_players():
            p.treatment = next(treatments)

class Group(BaseGroup):
    pass


class Player(BasePlayer):
    treatment = models.StringField()

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
            [1, 'Deutschland'],
            [2, 'anderes EU-Land'],
            [3, 'nicht EU-Land'],
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

    field_of_work = models.IntegerField(
        label = "In welchem Bereich sind sie beruflich tätig",
        choices = [
            [1, 'Land- und Forstwirtschaft, Fischerei'],
            [2, 'Verarbeitendes Gewerbe'],
            [3, 'Energie- und Wasserversorgung'],
            [4, 'Baugewerbe'],
            [5, 'Handel'],
            [6, 'Gastgewerbe'],
            [7, 'Finanz- und Versicherungsdienstleistungen'],
            [8, 'Information und Kommunikation'],
            [9, 'Kunst, Unterhaltung und Erholung'],
            [10, 'Gesundheits und Sozialwesen'],
            [11, 'Erziehung und Unterricht'],
            [12, 'Erbringung von sonstigen Dienstleistungen'],
            [13, 'Anderes'],
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )


    """ Items of the Belief in Zero-Sum Game (BZSG) scale questionnaire.
    Taken from: Różycka-Tran, J., Boski, P., & Wojciszke, B. (2015).
    Belief in a zero-sum game as a social axiom: A 37-nation study. Journal of
    Cross-Cultural Psychology, 46(4), 525-548.

    """

    bzsg_1 = seven_options_likert_scale(questions.get("bzsg")[0])
    bzsg_2 = seven_options_likert_scale(questions.get("bzsg")[1])
    bzsg_3 = seven_options_likert_scale(questions.get("bzsg")[2])
    bzsg_4 = seven_options_likert_scale(questions.get("bzsg")[3])
    bzsg_5 = seven_options_likert_scale(questions.get("bzsg")[4])
    bzsg_6 = seven_options_likert_scale(questions.get("bzsg")[5])
    bzsg_7 = seven_options_likert_scale(questions.get("bzsg")[6])
    bzsg_8 = seven_options_likert_scale(questions.get("bzsg")[7])

    # Perceptions on redistribution questions

    redistribution_1 = seven_options_likert_scale(questions.get("redistribution")[0])
    redistribution_2 = seven_options_likert_scale(questions.get("redistribution")[1])
    redistribution_3 = seven_options_likert_scale(questions.get("redistribution")[2])
    redistribution_4 = seven_options_likert_scale(questions.get("redistribution")[3])

    # Vignette questions.
    random_question_1 = seven_options_likert_scale("")
    random_question_2 = seven_options_likert_scale("")
    random_question_3 = models.IntegerField(
        blank = False,
    )
