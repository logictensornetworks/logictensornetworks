from __future__ import annotations
from pyparsing import (alphanums, alphas, nums, delimitedList, Forward,
            Group, Keyword, Literal, opAssoc, infixNotation, Suppress, Word)
from functools import reduce
import dataclasses

import ltn
from ltn.wrapper.grounding import Grounding, OperatorConfig


@dataclasses.dataclass
class OperatorReferences:
    """Names for matching operators"""
    exists = "exists"
    forall = "forall"
    implies = "->"
    or_ = "|"
    and_ = "&"
    not_ = "~"


TREE_TAB = 2*" "

@dataclasses.dataclass
class OpNode:
    operator: str
    operands: list[AtomNode | OpenFormulaNode | BoundFormulaNode]

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"operator:{self.operator}")
        print(TREE_TAB*level+"operands:")
        level+=1
        for operand in self.operands:
            operand.print_tree(level=level)

    def eval(self, grounding: Grounding, op_config: OperatorConfig, **kwargs) -> ltn.Formula:
        operands = [x.eval(grounding=grounding, op_config=op_config) for x in self.operands]
        # Unary operator
        if self.operator == OperatorReferences.not_:
            return op_config.not_(operands[0])
        # Binary operator
        else:
            if self.operator == OperatorReferences.and_:
                bin_op = op_config.and_
            elif self.operator == OperatorReferences.or_:
                bin_op = op_config.or_
            else:
                bin_op = op_config.implies
            if len(operands) == 2:
                return bin_op(operands[0], operands[1])
            else:
                if self.operator == OperatorReferences.and_ and op_config.and_aggreg is not None:
                    return op_config.and_aggreg(operands)
                elif self.operator == OperatorReferences.or_ and op_config.or_aggreg is not None:
                    return op_config.or_aggreg(operands)
                else:
                    return reduce(bin_op, operands)
                

@dataclasses.dataclass
class BoundFormulaNode:
    prefix: QuantifierPrefixNode
    open_formula: OpenFormulaNode

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+"prefix:")
        self.prefix.print_tree(level=level+1)
        print(TREE_TAB*level+"open_formula:")
        self.open_formula.print_tree(level=level+1)

    def eval(self, grounding: Grounding, op_config: OperatorConfig, **kwargs
             ) -> ltn.Formula:
        quantifier, variables = self.prefix.eval(grounding=grounding, op_config=op_config)
        return quantifier(variables, self.open_formula.eval(grounding, op_config=op_config))
    

@dataclasses.dataclass
class QuantifierPrefixNode:
    quantifier: str
    variables: list[VarOrCstNode]

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"quantifier:{self.quantifier}")
        print(TREE_TAB*level+"variables:")
        level+=1
        for var in self.variables:
            var.print_tree(level=level)

    def eval(self, grounding: Grounding, op_config: OperatorConfig, **kwargs
             ) -> tuple[ltn.Wrapper_Quantifier, list[ltn.Variable]]:
        if self.quantifier == OperatorReferences.forall:
            quant = op_config.forall
        else:
            quant = op_config.exists
        variables = [x.eval(grounding=grounding, op_config=op_config) for x in self.variables]
        return quant, variables


@dataclasses.dataclass
class OpenFormulaNode:
    val: AtomNode | BoundFormulaNode

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        self.val.print_tree(level=level)

    def eval(self, grounding: Grounding, **kwargs) -> ltn.Formula:
        return self.val.eval(grounding=grounding, **kwargs)


@dataclasses.dataclass
class FuncTermNode:
    function: FuncNode
    arguments: list[VarOrCstNode | FuncTermNode | NumberNode]

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+"function:")
        self.function.print_tree(level=level+1)
        print(TREE_TAB*level+"args:")
        for argument in self.arguments:
            argument.print_tree(level=level+1)

    def eval(self, grounding: Grounding, **kwargs) -> ltn.Term:
        func = self.function.eval(grounding, **kwargs)
        arg_terms = [a.eval(grounding, **kwargs) for a in self.arguments]
        return func(arg_terms)



@dataclasses.dataclass
class AtomNode:
    predicate: PredNode
    arguments: list[VarOrCstNode | FuncTermNode | NumberNode]

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+"pred:")
        self.predicate.print_tree(level=level+1)
        print(TREE_TAB*level+"args:")
        for argument in self.arguments:
            argument.print_tree(level=level+1)

    def eval(self, grounding: Grounding, **kwargs) -> ltn.Formula:
        pred = self.predicate.eval(grounding, **kwargs)
        arg_terms = [a.eval(grounding, **kwargs) for a in self.arguments]
        return pred(arg_terms)


@dataclasses.dataclass
class VarOrCstNode:
    val: str
    
    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"val:{self.val}")

    def eval(self, grounding: Grounding, **kwargs) -> (ltn.Variable | ltn.Constant):
        try:
            return grounding.variables[self.val]
        except KeyError:
            return grounding.constants[self.val]

@dataclasses.dataclass
class PredNode:
    val: str

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"val:{self.val}")
    
    def eval(self, grounding: Grounding, **kwargs) -> ltn.Predicate:
        return grounding.predicates[self.val]

@dataclasses.dataclass
class FuncNode:
    val: str

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"val:{self.val}")

    def eval(self, grounding: Grounding, **kwargs) -> ltn.Function:
        return grounding.functions[self.val]
    

@dataclasses.dataclass
class NumberNode:
    val: int

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"val:{self.val}")

    def eval(self, grounding: Grounding, **kwargs) -> ltn.Constant:
        return ltn.Constant(self.val, trainable=False)


@dataclasses.dataclass
class PropositionalVarNode:
    val: str

    def print_tree(self, level: int = 0) -> str:
        print(TREE_TAB*level+f"<{self.__class__.__name__}>")
        level+=1
        print(TREE_TAB*level+f"val:{self.val}")

    def eval(self, grounding: Grounding, **kwargs) -> ltn.Proposition:
        return grounding.propositions[self.val]


class Parser:
    def __init__(self) -> None:
        left_parenthesis, right_parenthesis = map(Suppress, "()")
        self.exists = Keyword(OperatorReferences.exists)
        self.forall = Keyword(OperatorReferences.forall)
        self.implies = Literal(OperatorReferences.implies)
        self.or_ = Literal(OperatorReferences.or_)
        self.and_ = Literal(OperatorReferences.and_)
        self.not_ = Literal(OperatorReferences.not_)

        var_symbol = Word( alphas+"_"+"?", alphanums+"_").set_parse_action(lambda tokens: VarOrCstNode(val=tokens[0]))
        func_symbol = Word( alphas+"_"+"?", alphanums+"_").set_parse_action(lambda tokens: FuncNode(val=tokens[0]))
        number = Word( nums ).set_parse_action(lambda tokens: NumberNode(val=int(tokens[0])))


        # term declaration
        self.term = Forward()
        func_term = Group(func_symbol + left_parenthesis + delimitedList(self.term) + right_parenthesis)
        func_term.set_parse_action(lambda tokens: FuncTermNode(function=tokens[0][0], arguments=tokens[0][1:]))
        self.term << (func_term | var_symbol | number)

        # formula declaration
        propositional_var_symbol = Word( alphas+"_", alphanums+"_").set_parse_action(lambda tokens: PropositionalVarNode(val=tokens[0]))

        pred_symbol = Word( alphas+"_", alphanums+"_").set_parse_action(lambda tokens: PredNode(val=tokens[0]))
        atom = Forward()
        atom << Group(pred_symbol  + left_parenthesis + delimitedList(self.term) + right_parenthesis)
        atom.set_parse_action(lambda tokens: AtomNode(predicate=tokens[0][0], arguments=tokens[0][1:]))

        self.formula = Forward()
        parse_action_quantifier_prefix = lambda tokens: QuantifierPrefixNode(quantifier=tokens[0][0], variables=tokens[0][1:])
        parse_action_open_formula = lambda tokens: OpenFormulaNode(val=tokens[0][0])
        parse_action_bound_formula = lambda tokens: BoundFormulaNode(prefix=tokens[0], open_formula=tokens[1])
        forall_prefix = Group((left_parenthesis + self.forall + delimitedList(var_symbol, delim=",") + right_parenthesis)
                       | (self.forall + delimitedList(var_symbol, delim=",")) )
        forall_prefix.set_parse_action(parse_action_quantifier_prefix)
        exists_prefix = Group((left_parenthesis + self.exists + delimitedList(var_symbol, delim=",") + right_parenthesis)
                       | (self.exists + delimitedList(var_symbol, delim=",")) )
        exists_prefix.set_parse_action(parse_action_quantifier_prefix)
        open_formula =  Group((left_parenthesis + self.formula + right_parenthesis) | self.formula )
        open_formula.set_parse_action(parse_action_open_formula)
        forall_expression = forall_prefix + open_formula
        forall_expression.set_parse_action(parse_action_bound_formula)
        exists_expression = exists_prefix + open_formula
        exists_expression.set_parse_action(parse_action_bound_formula)
        operand = forall_expression | exists_expression | atom | propositional_var_symbol

        parse_action_unary_op = lambda tokens: OpNode(operator=tokens[0][0], operands=[tokens[0][1]])
        parse_action_binary_op = lambda tokens: OpNode(operator=tokens[0][1], operands=tokens[0][::2])
        self.formula << infixNotation(operand,[(self.not_, 1, opAssoc.RIGHT, parse_action_unary_op),
                                            (self.and_, 2, opAssoc.LEFT, parse_action_binary_op),
                                            (self.or_, 2, opAssoc.LEFT, parse_action_binary_op),
                                            (self.implies, 2, opAssoc.RIGHT, parse_action_binary_op)])
        
    def parse_term(self, text: str) -> list:
        """ 
        Returns the parse results as a nested list of matching tokens, all converted to strings. 
        """
        result = self.term.parse_string(text, parse_all=True)
        return result
    
    def parse_formula(self, text: str) -> list:
        """
        
        """
        result = self.formula.parse_string(text, parse_all=True)
        return result



