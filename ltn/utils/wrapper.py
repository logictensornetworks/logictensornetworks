from __future__ import annotations
import ltn

def get_product_real_logic_config(p_exists: int = 5) -> ltn.wrapper.OperatorConfig:
    return ltn.wrapper.OperatorConfig(
        not_ = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std()),
        and_ = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod()),
        or_ = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum()),
        implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach()),
        exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=p_exists), semantics="exists"),
        forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_LogProd(),semantics="forall"),
        and_aggreg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Prod()),
        or_aggreg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMean(p=p_exists))
    )

def get_stable_operator_config(p_forall: int = 2, p_exists: int = 5) -> ltn.wrapper.OperatorConfig:
    return ltn.wrapper.OperatorConfig(
        not_ = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std()),
        and_ = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod()),
        or_ = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum()),
        implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach()),
        exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=p_exists), semantics="exists"),
        forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=p_forall),semantics="forall"),
        and_aggreg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_Prod()),
        or_aggreg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMean(p=p_exists))
    )