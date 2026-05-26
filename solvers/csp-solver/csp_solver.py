import func_timeout
import re
from collections import defaultdict

class CSP_Program:
    def __init__(self, logic_program: str, dataset_name: str) -> None:
        self.logic_program = logic_program
        self.flag = self.parse_logic_program()
        self.dataset_name = dataset_name
        self.timeout = 20

    def parse_logic_program(self):
        keywords = ['Query:', 'Constraints:', 'Variables:', 'Domain:']
        program_str = self.logic_program
        for keyword in keywords:
            try:
                program_str, segment_list = self._parse_segment(program_str, keyword)
                setattr(self, keyword[:-1], segment_list)
            except:
                setattr(self, keyword[:-1], None)

        if self.Query is None or self.Constraints is None or self.Variables is None or self.Domain is None:
            return False
        else:
            return True

    def _parse_segment(self, program_str, key_phrase):
        remain_program_str, segment = program_str.split(key_phrase)
        segment_list = segment.strip().split('\n')
        for i in range(len(segment_list)):
            segment_list[i] = segment_list[i].split(':::')[-1].strip()
        return remain_program_str, segment_list

    def safe_execute(self, code_string: str, keys=None, debug_mode=False):
        def execute(x):
            try:
                exec(x)
                locals_ = locals()
                if keys is None:
                    return locals_.get('ans', None), ""
                else:
                    return [locals_.get(k, None) for k in keys], ""
            except Exception as e:
                if debug_mode:
                    print(e)
                return None, e

        try:
            ans, error_msg = func_timeout.func_timeout(self.timeout, execute, args=(code_string,))
        except func_timeout.FunctionTimedOut:
            ans = None
            error_msg = "timeout"

        return ans, error_msg

    # comparison (>, <), fixed value (==, !=), etc
    def parse_numeric_constraint(self, constraint):
        # get all the variables in the rule from left to right
        pattern = r'\b[a-zA-Z_]+\b'  # Matches word characters (letters and underscores)
        variables_in_rule = re.findall(pattern, constraint)
        unique_list = []
        for item in variables_in_rule:
            if item not in unique_list:
                unique_list.append(item)
        str_variables_in_rule = ', '.join(unique_list)
        str_variables_in_rule_with_quotes = ', '.join([f'"{v}"' for v in unique_list]) + ','
        parsed_constraint = f"lambda {str_variables_in_rule}: {constraint}, ({str_variables_in_rule_with_quotes})"
        return parsed_constraint

    # all different constraint
    def parse_all_different_constraint(self, constraint):
        pattern = r'AllDifferentConstraint\(\[(.*?)\]\)'
        # Extract the content inside the parentheses
        result = re.search(pattern, constraint)
        if result:
            values_str = result.group(1)
            values = [value.strip() for value in values_str.split(',')]
        else:
            return None
        parsed_constraint = f"AllDifferentConstraint(), {str(values)}"
        return parsed_constraint

    def execute_program(self, debug_mode=False):
        # parse the logic program into CSP python program
        python_program_list = ['from constraint import *', 'problem = Problem()']
        # add variables
        if self.Variables:
            for variable in self.Variables:
                variable_name, variable_domain = variable.split('[IN]')
                variable_name, variable_domain = variable_name.strip(), variable_domain.strip()
                # variable_domain = ast.literal_eval(variable_domain)
                python_program_list.append(f'problem.addVariable("{variable_name}", {variable_domain})')

        # add constraints
        for rule in self.Constraints:
            rule = rule.strip()
            parsed_constraint = None
            if rule.startswith('AllDifferentConstraint'):
                parsed_constraint = self.parse_all_different_constraint(rule)
            else:
                parsed_constraint = self.parse_numeric_constraint(rule)
            # create the constraint
            python_program_list.append(f'problem.addConstraint({parsed_constraint})')

        # solve the problem
        python_program_list.append(f'ans = problem.getSolutions()')
        # execute the python program
        py_program_str = '\n'.join(python_program_list)
        if debug_mode:
            print(py_program_str)

        ans, err_msg = self.safe_execute(py_program_str, debug_mode=debug_mode)
        return ans, err_msg

    def answer_mapping(self, answer):
        self.option_pattern = r'^\w+\)'
        self.expression_pattern = r'\w+ == \d+'

        variable_ans_map = defaultdict(set)
        if len(answer) == 2:
            answer = answer[0]
        if not answer:
            answer = ""
        for result in answer:
            for variable, value in result.items():
                variable_ans_map[variable].add(value)

        print('VarAns Map-----', variable_ans_map)
        option_list = []

        for option_str in self.Query:
            # Extract the option using regex
            option_match = re.match(self.option_pattern, option_str)
            option = option_match.group().replace(')', '')
            # Extract the expression using regex
            expression_match = re.search(self.expression_pattern, option_str)
            expression_str = expression_match.group()
            # Extract the variable and its value
            variable, value = expression_str.split('==')
            variable, value = variable.strip(), int(value.strip())
            # Check if the variable is in the execution result
            if len(variable_ans_map[variable]) == 1 and value in variable_ans_map[variable]:
                option_list.append(option)

        if len(option_list) > 1 or len(option_list) == 0:
            return None, len(answer)
        return option_list[0], len(answer)


if __name__ == "__main__":
    logic_program = "Domain:\n1: leftmost \n 5: rightmost\nVariables:\ncardinal [IN] [1, 2, 3, 4, 5] \n crow [IN] [1, 2, 3, 4, 5] \n falcon [IN] [1, 2, 3, 4, 5] \n robin [IN] [1, 2, 3, 4, 5] \n blue_jay [IN] [1, 2, 3, 4, 5]\nConstraints:\nThe robin is to the right of the cardinal. ::: robin > cardinal \n The cardinal is to the right of the blue jay. ::: cardinal > blue_jay \n The blue jay is the second from the left. ::: blue_jay == 2 \n The crow is the rightmost. ::: crow == 5 \n All birds have different values. ::: AllDifferentConstraint([cardinal, crow, falcon, robin, blue_jay])\n\nQuery:\nThe cardinal is the second from the left. ::: A) cardinal == 2 \n The crow is the second from the left. ::: B) crow == 2 \n The falcon is the second from the left. ::: C) falcon == 2 \n The robin is the second from the left. ::: D) robin == 2 \n The blue jay is the second from the left. ::: E) blue_jay == 2"
    logic_program2 = "Domain:\n1: leftmost \n 5: rightmost\nVariables:\ncardinal [IN] [1, 2, 3, 4, 5] \n crow [IN] [1, 2, 3, 4, 5] \n falcon [IN] [1, 2, 3, 4, 5] \n robin [IN] [1, 2, 3, 4, 5] \n blue_jay [IN] [1, 2, 3, 4, 5]\nConstraints:\nThe robin is to the right of the cardinal. ::: robin < cardinal\n The cardinal is to the right of the blue jay. ::: cardinal > blue_jay \n The blue jay is the second from the left. ::: blue_jay == 2 \n The crow is the rightmost. ::: crow != 5\n All birds have different values. ::: AllDifferentConstraint([cardinal, crow, falcon, robin, blue_jay])\n\nQuery:\nThe cardinal is the second from the left. ::: A) cardinal == 2 \n The crow is the second from the left. ::: B) crow == 2 \n The falcon is the second from the left. ::: C) falcon == 2 \n The robin is the second from the left. ::: D) robin == 2 \n The blue jay is the second from the left. ::: E) blue_jay == 2"

    csp_program = CSP_Program(logic_program, 'LogicalDeduction')
    ans = csp_program.execute_program()
    csp_program2 = CSP_Program(logic_program2, 'LogicalDeduction')
    ans2 = csp_program2.execute_program()
    print(ans)
    print(ans2)
    print(csp_program.answer_mapping(ans))
    print(csp_program.answer_mapping(ans2))