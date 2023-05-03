import random

from nmmo.task.predicate import Predicate, AND, NOT, OR

# TODO: type hints
# TODO: include NOT
# TODO: include some way to indicate subject/target being same/other teams, more particular agents etc;
#       right now only subject=current_team is supported
# TODO: better structure for `clauses`

class RandomTaskGenerator:
  def __init__(self) -> None:
    """
    Generates Tasks by randomly combining Predicates
    """
    self._pred_specs = []
    self._pred_spec_weights = []

  def add_pred_spec(self, pred_class, param_space = None, weight: float = 1):
    """
    Builds the list of Predicates to choose from when sampling

    Args:
      pred_class: (base) Predicate class
      param_space: list of lists containing options for each param to be randomly selected;
                   should not include a list for the 'subject' param as that is defined at runtime
      weight: weighting for this pred_spec in random choice
    """
    self._pred_specs.append((pred_class, param_space or []))
    self._pred_spec_weights.append(weight)

  def sample(self,
             min_clauses: int = 1,
             max_clauses: int = 1,
             min_clause_size: int = 1,
             max_clause_size: int = 1,
             not_p: float = 0.0) -> Predicate:
    """
    Generate a task by combining Predicates from self._pred_specs into clauses
    with AND/OR and putting those clauses together with an AND/OR
        
    Args:
        min_clauses: min clauses in the task
        max_clauses: max clauses in the task
        min_clause_size: min Predicates in each clause
        max_clause_size: max Predicates in each clause
        not_p: probability that a Predicate will be NOT'd
    """


    # first index contains combiner class for all the clauses
    # the rest of the indices contain the clauses
    clauses = []
    if random.random() < .5:
        clauses.append(AND)
    else:
        clauses.append(OR)

    # iterate once for each clause
    for _ in range(0, random.randint(min_clauses, max_clauses)):
      pred_specs = random.choices(
        self._pred_specs,
        weights = self._pred_spec_weights,
        k = random.randint(min_clause_size, max_clause_size)
      )

      # first index contains combiner class for all the predicates in this clause
      # the rest of the indices contain lists with those predicates and their params
      pred_list = []
      if random.random() < .5:
          pred_list.append(AND)
      else:
          pred_list.append(OR)
      for pred_class, pred_param_space in pred_specs:
        pred_list.append([pred_class, *[random.choice(pp) for pp in pred_param_space]])

      clauses.append(pred_list)
    
    return clauses
