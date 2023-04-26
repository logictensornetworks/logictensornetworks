from pyparsing import (alphanums, alphas, nums, delimitedList, Forward,
            Group, Keyword, Literal, opAssoc, infixNotation,
            ParserElement, ParseException, ParseSyntaxException, Suppress,
            Word)
import dataclasses
import json

@dataclasses.dataclass
class TokenReferences:
    """Names for referencing matching tokens"""
    TERM = "term"
    LITERAL = "literal"
    VAR = "var"
    VAR_OR_CST = "var_or_cst"
    NUMBER = "number"
    PRED = "pred"
    FUNC = "func"
    FORMULA = "formula"
    OPERATOR = "op"
    QUANTIFIER = "agg"
    OPENFORMULA = "openformula"
    ARGS = "args"


class Parser:
    def __init__(self) -> None:
        left_parenthesis, right_parenthesis = map(Suppress, "()")
        self.exists = Keyword("exists")
        self.forall = Keyword("forall")
        self.implies = Literal("->")
        self.or_ = Literal("|")
        self.and_ = Literal("&")
        self.not_ = Literal("~")

        symbol = Word( alphas+"_"+"?", alphanums+"_")
        number = Word( nums )

        tr = TokenReferences()

        # term declaration
        self.term = Forward()
        self.term << (Group(symbol(tr.FUNC) + Group(left_parenthesis + delimitedList(self.term) + right_parenthesis)(tr.ARGS)).set_results_name(tr.TERM,list_all_matches=True) 
                | symbol.set_results_name(tr.VAR_OR_CST, list_all_matches=True) 
                | number.set_results_name(tr.NUMBER, list_all_matches=True))

        # formula declaration
        pred_symbol = Word( alphas+"_", alphanums+"_")
        literal = Forward()
        literal << (Group(pred_symbol(tr.PRED)  + Group(left_parenthesis + delimitedList(self.term) + right_parenthesis)(tr.ARGS)) | 
                  Group(self.not_ + pred_symbol(tr.PRED)  + Group(left_parenthesis + delimitedList(self.term) + right_parenthesis)(tr.ARGS)))
        self.formula = Forward()
        forall_expression = Group(self.forall(tr.QUANTIFIER) + delimitedList(symbol.set_results_name(tr.VAR, list_all_matches=True), delim=",") 
                                + (Group(left_parenthesis + self.formula + right_parenthesis)(tr.OPENFORMULA) | literal(tr.LITERAL) ))
        exists_expression = Group(self.exists(tr.QUANTIFIER) + delimitedList(symbol.set_results_name(tr.VAR, list_all_matches=True), delim=",") 
                                + (Group(left_parenthesis + self.formula + right_parenthesis)(tr.OPENFORMULA) | literal(tr.LITERAL) ))
        operand = forall_expression(tr.FORMULA) | exists_expression(tr.FORMULA) | literal(tr.LITERAL)

        self.formula << infixNotation(operand,[(self.not_, 1, opAssoc.RIGHT),
                                            (self.and_, 2, opAssoc.LEFT),
                                            (self.or_, 2, opAssoc.LEFT),
                                            (self.implies, 2, opAssoc.RIGHT)])
        
    def parse_term(self, text: str) -> list:
        """ 
        Returns the parse results as a nested list of matching tokens, all converted to strings. 
        """
        result = self.term.parse_string(text, parse_all=True)
        return result.as_dict()
    
    def parse_formula(self, text: str) -> list:
        """
        >>> formula = "p(a,b)"
        >>> print(parse_string(formula))
        ['p', (['a', 'b'], {})]

        >>> formula = "~p(a,b)"
        >>> print(parse_string(formula))
        ['~','p', (['a', 'b'], {})]

        >>> formula = "=(a,b)"
        >>> print(parse_string(formula))
        ['=', (['a', 'b'], {})]

        >>> formula = "<(a,b)"
        >>> print(parse_string(formula))
        ['<', (['a', 'b'], {})]
        
        >>> formula = "~p(a)"
        >>> print(parse_string(formula))    
        ['~', 'p', (['a'], {})]

        >>> formula = "~p(a)|a(p)"
        >>> print(parse_string(formula))
        [(['~', 'p', (['a'], {})], {}), '|', (['a', (['p'], {})], {})]

        >>> formula = "p(a) | p(b)"
        >>> print(parse_string(formula))
        [(['p', (['a'], {})], {}), '|', (['p', (['b'], {})], {})]

        >>> formula = "~p(a) | p(b)"
        >>> print(parse_string(formula))
        [(['~', 'p', (['a'], {})], {}), '|', (['p', (['b'], {})], {})]

        >>> formula = "p(f(a)) | p(b)"
        >>> print(parse_string(formula))
        [(['p', ([(['f', (['a'], {})], {})], {})], {}), '|', (['p', (['b'], {})], {})]

        >>> formula = "p(a) | p(b) | p(c)"
        >>> print(parse_string(formula))
        [(['p', ([(['f', (['a'], {})], {})], {})], {}), '|', (['p', (['b'], {})], {})]

        
        Note that
            forall x P(x) & Q(x) 
        is interpreted as
            (forall x P(x)) & Q(x) 
        """
        result = self.formula.parse_string(text, parse_all=True)
        return result.as_dict()




parser = Parser()

#parse_results = parser.parse_formula("forall disease ( exists drug (cures(drug,disease)) )")
#print(parse_results)

# print(parser.parse_formula("p(a,b)"))
# print(parser.parse_formula("forall x,y P(x,y)"))
#parse_results = parser.parse_formula("forall x ( exists y P(x,y) )")

parse_results = parser.parse_formula("A(x) & B(x)")
# print(parser.parse_formula("p(f(a,b),c,f(d,f(a)))"))
print(json.dumps(parse_results, indent=2))
