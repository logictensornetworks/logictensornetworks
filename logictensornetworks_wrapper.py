try:
    from pyparsing import (alphanums, alphas, delimitedList, Forward,
            Group, Keyword, Literal, opAssoc, operatorPrecedence,
            ParserElement, ParseException, ParseSyntaxException, Suppress,
            Word)
except ImportError:
    from pyparsing_py3 import (alphanums, alphas, delimitedList, Forward,
            Group, Keyword, Literal, opAssoc, operatorPrecedence,
            ParserElement, ParseException, ParseSyntaxException, Suppress,
            Word)
ParserElement.enablePackrat()

import tensorflow as tf
import logictensornetworks as ltn
import logging

CONFIGURATION = { "max_nr_iterations" : 1000, "error_on_redeclare" : False}

CONSTANTS={}
PREDICATES={}
VARIABLES={}
FUNCTIONS={}
TERMS={}
FORMULAS={}
AXIOMS={}

def set_tnorm(tnorm):
    ltn.set_tnorm(tnorm)
    
def set_universal_aggreg(aggr):
    ltn.set_universal_aggreg(aggr)

def set_existential_aggregator(aggr):
    ltn.set_existential_aggregator(aggr)
def set_layers(layers):
    ltn.LAYERS=layers

def constant(label,*args,**kwargs):
    if label in CONSTANTS and args==() and kwargs=={}:
        return CONSTANTS[label]
    elif label in CONSTANTS and CONFIGURATION.get("error_on_redeclare"):
        logging.getLogger(__name__).error("Attempt at redeclaring existing constant %s" % label)
        raise Exception("Attempt at redeclaring existing constant %s" % label)
    else:
        if label in CONSTANTS:
            logging.getLogger(__name__).warn("Redeclaring existing constant %s" % label)
        CONSTANTS[label]=ltn.constant(label,*args,**kwargs)
        return CONSTANTS[label]

def _variable_label(label):
    try:
        if label.startswith("?") and len(label) > 1:
            return "var_" + label[1:]
    except:
        pass
    return label

def variable(label,*args,**kwargs):
    label=_variable_label(label)
    if label in VARIABLES and args==() and kwargs=={}:
        return VARIABLES[label]
    elif label in VARIABLES and CONFIGURATION.get("error_on_redeclare"):
        logging.getLogger(__name__).error("Attempt at redeclaring existing variable %s" % label)
        raise Exception("Attempt at redeclaring existing variable %s" % label)
    else:
        if label in VARIABLES:
            logging.getLogger(__name__).warn("Redeclaring existing variable %s" % label)
        VARIABLES[label]=ltn.variable(label,*args,**kwargs)
        return VARIABLES[label]

def predicate(label,*args,**kwargs):
    if label in PREDICATES and args==() and kwargs=={}:
        return PREDICATES[label]
    elif label in PREDICATES and CONFIGURATION.get("error_on_redeclare"):
        logging.getLogger(__name__).error("Attempt at redeclaring existing predicate %s" % label)
        raise Exception("Attempt at redeclaring existing predicate %s" % label)
    else:
        if label in PREDICATES:
            logging.getLogger(__name__).warn("Redeclaring existing predicate %s" % label)
        PREDICATES[label]=ltn.predicate(label,*args,**kwargs)
        return PREDICATES[label]

def function(label,*args,**kwargs):
    if label in FUNCTIONS and args==() and kwargs=={}:
        return FUNCTIONS[label]
    elif label in FUNCTIONS and CONFIGURATION.get("error_on_redeclare"):
        logging.getLogger(__name__).error("Attempt at redeclaring existing function %s" % label)
        raise Exception("Attempt at redeclaring existing function %s" % label)
    else:
        if label in FUNCTIONS:
            logging.getLogger(__name__).warn("Redeclaring existing function %s" % label)
        FUNCTIONS[label]=ltn.function(label,*args,**kwargs)
        return FUNCTIONS[label]

def _parse_term(text):
    """ """
    left_parenthesis, right_parenthesis, colon = map(Suppress, "():")

    symbol = Word( alphas+"_"+"?"+".", alphanums+"_"+"?"+"."+"-")
    
    term = Forward()
    term << (Group(symbol + Group(left_parenthesis +
                   delimitedList(term) + right_parenthesis)) | symbol)

    
    result = term.parseString(text, parseAll=True)
    
    return result.asList()[0]

OPERATORS={"|" : ltn.Or,
           "&" : ltn.And,
           "~" : ltn.Not,
           "->" : ltn.Implies,
           "%" : ltn.Equiv}

def _parse_formula(text):
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

    """
    left_parenthesis, right_parenthesis, colon = map(Suppress, "():")
    exists = Keyword("exists")
    forall = Keyword("forall")
    implies = Literal("->")
    or_ = Literal("|")
    and_ = Literal("&")
    not_ = Literal("~")
    equiv_ = Literal("%")
    

    symbol = Word( alphas+"_"+"?"+".", alphanums+"_"+"?"+"."+"-")
    
    term = Forward()
    term << (Group(symbol + Group(left_parenthesis +
                   delimitedList(term) + right_parenthesis)) | symbol)

    pred_symbol = Word( alphas+"_"+".", alphanums+"_"+"."+"-") | Literal("=") | Literal("<")
    literal = Forward()
    literal << (Group(pred_symbol  + Group(left_parenthesis + delimitedList(term) + right_parenthesis)) | 
                  Group(not_ + pred_symbol  + Group(left_parenthesis + delimitedList(term) + right_parenthesis)))

    formula = Forward()
    forall_expression = Group(forall + delimitedList(symbol) + colon + formula)
    exists_expression = Group(exists + delimitedList(symbol) + colon + formula)
    operand = forall_expression | exists_expression | literal

    formula << operatorPrecedence(operand,[(not_, 1, opAssoc.RIGHT),
                                           (and_, 2, opAssoc.LEFT),
                                           (or_, 2, opAssoc.LEFT),
                                           (equiv_, 2, opAssoc.RIGHT),
                                           (implies, 2, opAssoc.RIGHT)])
    result = formula.parseString(text, parseAll=True)
    
    return result.asList()[0]

def _build_term(term):
    try:
        if str(term) in CONSTANTS:
            return CONSTANTS[term]
    except:
        pass
    try:
        if str(_variable_label(term)) in VARIABLES:
            return VARIABLES[_variable_label(term)]
    except:
        pass
    try:
        if term[0] in FUNCTIONS:
            return FUNCTIONS[term[0]](*[_build_term(t) for t in term[1]])
    except:
        pass

    raise Exception("Could not build term for %s. Not a declared constant or variable. Also building it as a function failed." % str(term))

def term(term):
    global TERMS
    if term not in TERMS:
        TERMS[term]=_build_term(_parse_term(term))
    return TERMS[term]
 
def _build_formula(formula):
    if not isinstance(formula,list) or not len(formula)>1:
        raise Exception("Cannot build formula for %s" % str(formula))
    elif str(formula[0]) in PREDICATES:
        terms=[]
        for t in formula[1]:
            _t=_build_term(t)
            if _t is None:
                return None
            terms.append(_t)
        return PREDICATES[formula[0]](*terms)
    elif str(formula[0]) == "~":
        return ltn.Not(_build_formula(formula[1]))
    elif str(formula[0]) == "forall":
        variables=[]
        for t in formula[1:-1]:
            if not _variable_label(t) in VARIABLES:
                raise Exception("%s in %s not a variable" % (t,formula))
            variables.append(VARIABLES[_variable_label(t)])
        variables=tuple(variables)
        wff=_build_formula(formula[-1])
        return ltn.Forall(variables,wff)
    elif str(formula[0]) == "exists":
        variables=[]
        for t in formula[1:-1]:
            if not _variable_label(t) in VARIABLES:
                raise Exception("%s in %s not a variable" % (t,formula))
            variables.append(VARIABLES[_variable_label(t)])
        variables=tuple(variables)
        wff=_build_formula(formula[-1])
        return ltn.Exists(variables,wff)
    else:
        operator=None
        formulas=[]
        for c in formula:
            if str(c) in OPERATORS:
                assert(operator is None or c==operator)
                operator=c
            else:
                formulas.append(c)
        formulas=[_build_formula(f) for f in formulas]
        return OPERATORS[operator](*formulas)
    raise Exception("Unable to build formula for %s" % str(formula))

def formula(formula):
    global FORMULAS
    if formula not in FORMULAS:    
        FORMULAS[formula]=_build_formula(_parse_formula(formula))
    return FORMULAS[formula]

def axiom(axiom):
    global AXIOMS
    if axiom not in AXIOMS:
        AXIOMS[axiom]=formula(axiom)
    return AXIOMS[axiom]

def _compute_feed_dict(feed_dict):
    """ Maps constant and variable string in feed_dict 
        to their tensors """
    _feed_dict={}
    for k,v in feed_dict.items():
        if k in CONSTANTS:
            _feed_dict[CONSTANTS[k]]=v
        elif _variable_label(k) in VARIABLES:
            _feed_dict[VARIABLES[_variable_label(k)]]=v
        else:
            _feed_dict[k]=v
    return _feed_dict

SESSION=None
OPTIMIZER=None
KNOWLEDGEBASE=None

def initialize_tf_variables(all=False):
    if tf.global_variables():
        if all:
            init = tf.global_variables_initializer()
        else:
            uninitialized_variable_names=set([v.decode("ascii") for v in SESSION.run(tf.report_uninitialized_variables())])
            uninitialized_variables=[var for var in tf.global_variables() if var.name.split(":")[0] in uninitialized_variable_names]
            init=tf.variables_initializer(uninitialized_variables)
        SESSION.run(init)

def initialize_tf_session():
    """ initialize tf InteractiveSession. you can call this multiple times 
        initialize_all_variables -- this will initialize """
    global SESSION

    if SESSION:
        logging.getLogger(__name__).warn("Closing existing Tensorflow session.")
        SESSION.close()

    logging.getLogger(__name__).info("Initializing Tensorflow session")
    SESSION = tf.InteractiveSession()
    initialize_tf_variables(all=True)

def initialize_knowledgebase(optimizer=None,
    formula_aggregator=lambda *x: tf.reduce_mean(tf.concat(x,axis=0)) if x else None,
    initial_sat_level_threshold=0.0,
    track_sat_levels=10,
    max_trials=100,
    feed_dict={}):
    global OPTIMIZER,KNOWLEDGEBASE

    
    if AXIOMS.values():
        logging.getLogger(__name__).info("Initializing knowledgebase")
        KNOWLEDGEBASE=formula_aggregator(*AXIOMS.values()) 
    else:
        logging.getLogger(__name__).info("No axioms. Skipping knowledgebase aggregation")
    
    initialize_tf_session()
    
    # if there are variables to optimize
    if tf.global_variables() and KNOWLEDGEBASE is not None:

        logging.getLogger(__name__).info("Initializing optimizer")
        OPTIMIZER = optimizer or tf.train.GradientDescentOptimizer(learning_rate=0.1)
        OPTIMIZER=OPTIMIZER.minimize(-KNOWLEDGEBASE)

        logging.getLogger(__name__).info("Assembling feed dict")
        _feed_dict=_compute_feed_dict(feed_dict)
    
        init = tf.global_variables_initializer()
        sat_level = SESSION.run(KNOWLEDGEBASE,feed_dict=_feed_dict)
        i=0
        for i in range(max_trials):
            SESSION.run(init,feed_dict=_feed_dict)
            sat_level = SESSION.run(KNOWLEDGEBASE,feed_dict=_feed_dict)
            if  initial_sat_level_threshold is not None and sat_level >= initial_sat_level_threshold:
                break
            if track_sat_levels is not None and i % track_sat_levels == 0:
                logging.getLogger(__name__).info("INITIALIZE %s sat level -----> %s" % (i,sat_level))
        logging.getLogger(__name__).info("INITIALIZED with sat level = %s" % (sat_level))

def train(max_epochs=10000,
        track_sat_levels=100,
        sat_level_epsilon=.99,
        feed_dict={}):
    global SESSION,OPTIMIZER,KNOWLEDGEBASE
    if KNOWLEDGEBASE is None:
        raise Exception("KNOWLEDGEBASE not initialized. Please run initialize_knowledgebase first.")

    _feed_dict=_compute_feed_dict(feed_dict)

    if tf.global_variables():    
        for i in range(max_epochs):
            if track_sat_levels is not None and i % track_sat_levels == 0:
                sat_level=SESSION.run(KNOWLEDGEBASE,feed_dict=_feed_dict)
                logging.getLogger(__name__).info("TRAINING %s sat level -----> %s" % (i,sat_level))
                if sat_level_epsilon is not None and sat_level > sat_level_epsilon:
                    logging.getLogger(__name__).info("TRAINING finished after %s epochs with sat level %s" % (i,sat_level))
                    return sat_level
            
            SESSION.run(OPTIMIZER,feed_dict=_feed_dict)
        sat_level=SESSION.run(KNOWLEDGEBASE,feed_dict=_feed_dict)
        logging.getLogger(__name__).info("TRAINING finished after %s epochs with sat level %s" % (i,sat_level))
        return sat_level
    else:
        logging.getLogger(__name__).warn("Nothing to optimize/train. Skipping training")
        return SESSION.run(KNOWLEDGEBASE,feed_dict=_feed_dict)

def ask(term_or_formula,feed_dict={}):
    global SESSION
    if SESSION is None:
        initialize_tf_session()
    _t = None
    try:
        _t=_build_formula(_parse_formula(term_or_formula))
    except:
        pass
    try:
        _t=_build_term(_parse_term(term_or_formula))
    except:
        pass
    if _t is None:
        raise Exception('Could not parse and build term/formula for "%s"' % term_or_formula)
    else:
        _feed_dict=_compute_feed_dict(feed_dict)

        return SESSION.run(_t,feed_dict=_feed_dict)

def ask_m(terms_or_formulas,feed_dict={}):
    global SESSION
    if SESSION is None:
        initialize_tf_session()
    _ts=[]
    for term_or_formula in terms_or_formulas:
        _t = None
        try:
            _t=_build_formula(_parse_formula(term_or_formula))
        except:
            pass
        try:
            _t=_build_term(_parse_term(term_or_formula))
        except:
            pass
        if _t is None:
            raise Exception('Could not parse and build term/formula for "%s"' % term_or_formula)
        else:
            _ts.append(_t)
    _feed_dict=_compute_feed_dict(feed_dict)
    return SESSION.run(_ts,feed_dict=_feed_dict)

def _reset():
    global CONSTANTS,PREDICATES,VARIABLES,FUNCTIONS,TERMS,FORMULAS,AXIOMS
    global KNOWLEDGEBASE,SESSION,OPTIMIZER
    CONSTANTS={}
    PREDICATES={}
    VARIABLES={}
    FUNCTIONS={}
    FORMULAS={}
    AXIOMS={}
    KNOWLEDGEBASE=None
    if SESSION:
        SESSION.close()
    SESSION=None
    OPTIMIZER=None

