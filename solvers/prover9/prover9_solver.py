from nltk.inference.prover9 import *
from nltk.sem.logic import NegatedExpression
from .fol_prover9_parser import Prover9_FOL_Formula
from .Formula import FOL_Formula
import os

# set the path to the prover9 executable
os.environ['PROVER9'] = '../LADR-2009-11A/bin/prover9'

class FOL_Prover9_Program:
    def __init__(self, premise_program: list, conclusion_program: str, dataset_name='FOLIO') -> None:
        self.premise_program = premise_program
        self.conclusion_program = conclusion_program
        self.flag, self.parse_error = self.parse_logic_program()
        self.dataset_name = dataset_name

    def parse_logic_program(self):
        try:
            # Extract each premise and the conclusion using regex
            premises = self.premise_program
            conclusion = self.conclusion_program.strip()
            self.logic_premises = [premise.split(':::')[1].replace("\'", "").strip() for premise in premises if
                                   ':::' in premise]
            self.logic_conclusion = conclusion.split(':::')[1].strip('*').strip()

            # convert to prover9 format
            self.prover9_premises = []
            for premise in self.logic_premises:
                fol_rule = FOL_Formula(premise)
                if fol_rule.is_valid == False:
                    return False, f'FOL rule is invalid for {premise}'
                prover9_rule = Prover9_FOL_Formula(fol_rule)
                self.prover9_premises.append(prover9_rule.formula)

            fol_conclusion = FOL_Formula(self.logic_conclusion)
            if fol_conclusion.is_valid == False:
                return False, f'FOL rule is invalid for {self.logic_conclusion}'
            self.prover9_conclusion = Prover9_FOL_Formula(fol_conclusion).formula
            return True, ''
        except Exception as e:
            return False, f'Parse exception: {e}'

    def execute_program(self):
        try:
            goal = Expression.fromstring(self.prover9_conclusion)
            assumptions = [Expression.fromstring(a) for a in self.prover9_premises]
            timeout = 10

            prover = Prover9Command(goal, assumptions, timeout=timeout)
            result = prover.prove()  # fails to run on Mac
            # print(prover.proof())
            if result:
                return 'True', ''
            else:
                # If Prover9 fails to prove, we differentiate between False and Unknown
                # by running Prover9 with the negation of the goal
                negated_goal = NegatedExpression(goal)
                prover = Prover9Command(negated_goal, assumptions, timeout=timeout)
                negation_result = prover.prove()
                if negation_result:
                    return 'False', ''
                else:
                    return 'Unknown', ''
        except Exception as e:
            return None, str(e)

    def answer_mapping(self, answer):
        if answer == 'True':
            return 'A'
        elif answer == 'False':
            return 'B'
        elif answer == 'Unknown':
            return 'C'
        else:
            raise Exception("Answer not recognized")


if __name__ == "__main__":
    premise_program = [
        "Symphony No. 9 is a music piece. ::: MusicPiece(symphony9)",
        "Composers write music pieces. ::: ∀x∃y (MusicPiece(x) ∧ Write(y, x)) → Composer(y) ",
        "Beethoven wrote Symphony No. 9. ::: Write(beethoven, symphony9)",
        "Vienna Music Society premiered Symphony No. 9. ::: Premiered(viennaMusicSociety, symphony9)",
        "Vienna Music Society is an orchestra. ::: Orchestra(viennaMusicSociety)",
        "Beethoven leads the Vienna Music Society. ::: Lead(beethoven, viennaMusicSociety)",
        "Orchestras are led by conductors. ::: ∀x∀y ((Orchestra(x) ∧ Lead(y, x)) → Conductor(y))"
    ]
    conclusion_program = "Beethoven is not a conductor. ::: ¬Conductor(beethoven)"

    prover9_program = FOL_Prover9_Program(premise_program, conclusion_program)
    answer, error_message = prover9_program.execute_program()
    print('Program 1:', answer)
    print(answer, error_message)