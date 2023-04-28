
## Interpretation

### Associativity
In first-order logic (FOL), there is no universal rule for associativity that applies to all connectives or quantifiers. 
The associativity depends on the specific connectives used. To clarify the associativity for different connectives in FOL:
1. Conjunction (∧) and Disjunction (∨): if parenthesis are not provided, we interpret them as left-associative by convention. This means that:
    ```
    P ∧ Q ∧ R ≡ (P ∧ Q) ∧ R
    P ∨ Q ∨ R ≡ (P ∨ Q) ∨ R
    ```
2. Implication (→): by convention, they are treated as right-associative when parentheses are not provided:
    ```
    P → Q → R ≡ P → (Q → R)
    ```
### Operator Precedence
The order of precedence for the logical connectives, from highest to lowest, is as follows:
1. Negation (¬)
2. Conjunction (∧)
3. Disjunction (∨)
4. Implication (→)
For example, given the formula `P ∧ Q → R`, the order of precedence dictates that conjunction (∧) should be evaluated before implication (→). Therefore, the formula can be interpreted as:
`(P ∧ Q) → R`

### Quantifier Scope and Bound Variables
When you have multiple quantifiers (∀ and ∃) without parentheses, the scope of the quantifiers is determined by their order in the formula. Each quantifier applies to the formula that follows it, up to the end of the formula unless parentheses indicate otherwise. If multiple quantifiers are written sequentially without parentheses, they are typically understood to have the same scope. For example:

    ∀x ∃y P(x,y)∧ Q(x) ≡ (∀x)(∃y)(P(x,y)∧Q(x))

If multiple quantifiers of the same type are written sequentially without parentheses, like `∀x,y`, they are conventionally interpreted to have the same scope, applying to the entire formula that follows. 
For example:

    ∀x,y P(x,y) ≡ (∀x)(∀y) P(x,y)