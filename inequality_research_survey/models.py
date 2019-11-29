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
            [7, "1"],
            [6, "2"],
            [5, "3"],
            [4, "4"],
            [3, "5"],
            [2, "6"],
            [1, "7"],
        ],
        label = question,
        widget = widgets.RadioSelectHorizontal,
    )

class Constants(BaseConstants):
    name_in_url = 'inequality_research_survey'
    players_per_group = None
    num_rounds = 2


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
            # we need participant vars as we now have more than 1 round and normal variables are stored only within one round
            p.participant.vars['treatment'] = p.treatment

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
            [1, 'Deutsch'],
            [2, 'Eu-Bürger'],
            [3, 'Nicht-EU-Bürger'],
            # [1, 'Deutschland'],
            # [2, 'anderes EU-Land'],
            # [3, 'nicht EU-Land'],
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )

    education = models.IntegerField(
        label = "Was ist ihr höchster Bildungsabschluss?",
        choices = [
            [1, 'Hochschulabschluss'],
            [2, 'Berufsausbildung'],
            [3, 'Abitur'],
            [4, 'Realschulabschluss'],
            [5, 'Hauptschulabschluss'],
            [6, 'Keiner'],
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )

    field_of_work = models.IntegerField(
        label = "In welchem Bereich sind sie beruflich tätig?",
        choices = [
            [1, 'Land- und Forstwirtschaft, Fischerei'],
            [2, 'Verarbeitendes Gewerbe (Produktion, Handwerk...)'],
            [3, 'Energie- und Wasserversorgung'],
            [4, 'Baugewerbe'],
            [5, 'Handel'],
            [6, 'Gastgewerbe'],
            [7, 'Finanz- und Versicherungsdienstleistungen'],
            [8, 'Information und Kommunikation'],
            [9, 'Kunst, Unterhaltung und Erholung'],
            [10, 'Gesundheits- und Sozialwesen'],
            [11, 'Erziehung und Unterricht'],
            [12, 'Erbringung von sonstigen Dienstleistungen'],
            [13, 'SchülerIn/StudentIn'],
            [14, 'RentnerIn/PensionärIn'],
            [15, 'nicht berufstätig'],
        ],
        widget = widgets.RadioSelect,
        blank = False,
    )

    political_party = models.IntegerField(
        label = "Wenn am nächsten Sonntag Bundestagswahl wäre, welche der folgenden Parteien würden Sie dann wählen?",
        choices = [
            [1, "CDU/CSU"],
            [2, "SPD"],
            [3, "AfD"],
            [4, "FDP"],
            [5, "Die Linke"],
            [6, "Bündnis 90/Die Grünen"],
            [7, "Keine der oben genannten Parteien."],
            [8, "Ich würde nicht wählen gehen."],
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
    undefined_question_1 = models.IntegerField(
        choices = [
            [1, "sehr gerecht"],
            [2, "gerecht"],
            [3, "weder gerecht noch ungerecht"],
            [4, "ungerecht"],
            [5, "sehr ungerecht"],
        ],
            widget = widgets.RadioSelect,
            blank = False,
    )
    undefined_question_2 = models.IntegerField(
        choices = [
            [1, 'ja'],
            [0, 'nein'],
        ],
            widget = widgets.RadioSelect,
            blank = False,
    )
    undefined_question_3 = models.IntegerField(
        blank = True,
    )

    undefined_question_4 = models.IntegerField(
        blank = False,
    )

    zsg_question_1 = models.IntegerField(
        choices = [
            [1, "sehr gerecht"],
            [2, "gerecht"],
            [3, "weder gerecht noch ungerecht"],
            [4, "ungerecht"],
            [5, "sehr ungerecht"],
        ],
            widget = widgets.RadioSelect,
            blank = False,
    )
    zsg_question_2 = models.IntegerField(
        choices = [
            [1, 'ja'],
            [0, 'nein'],
        ],
            widget = widgets.RadioSelect,
            blank = False,
    )
    zsg_question_3 = models.IntegerField(
        blank = True,
    )

    zsg_question_4 = models.IntegerField(
        blank = False,
    )

    nzsg_question_1 = models.IntegerField(
        choices = [
            [1, "sehr gerecht"],
            [2, "gerecht"],
            [3, "weder gerecht noch ungerecht"],
            [4, "ungerecht"],
            [5, "sehr ungerecht"],
        ],
            widget = widgets.RadioSelect,
            blank = False,
    )
    nzsg_question_2 = models.IntegerField(
        choices = [
            [1, 'ja'],
            [0, 'nein'],
        ],
            widget = widgets.RadioSelect,
            blank = False,
    )
    nzsg_question_3 = models.IntegerField(
        blank = True,
    )

    nzsg_question_4 = models.IntegerField(
        blank = False,
    )
