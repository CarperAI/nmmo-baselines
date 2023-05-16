import random

from nmmo.task.predicate import Predicate, NOT

# TODO: type hints
# TODO: include some way to indicate subject/target being same/other teams, more particular agents etc;
#       right now only subject=current_team is supported
# TODO: better structure for `clauses`
# TODO: randomize reward for the task (include in info)

class RandomTaskGenerator:
  def __init__(self) -> None:
    """
    Generates info for clauses of Predicates which can be combined to create Tasks
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
    Randomly generates parameters that can be used to instantiate clauses of Predicates

    Args:
        min_clauses: min clauses in the task
        max_clauses: max clauses in the task
        min_clause_size: min Predicates in each clause
        max_clause_size: max Predicates in each clause
        not_p: probability that a Predicate will be NOT'd
    """

    # A list of lists of lists
    # outer list: each index corresponds to one clause
    # middle list: each index corresponds to one Predicate
    # inner list: contains the parameters used to instantiate this Predicate
    clauses = []

    # Iterate once for each clause
    for _ in range(random.randint(min_clauses, max_clauses)):
      pred_specs = random.choices(
        self._pred_specs,
        weights = self._pred_spec_weights,
        k = random.randint(min_clause_size, max_clause_size)
      )

      pred_list = [] # middle list
      for pred_class, pred_param_space in pred_specs:
        # Append the inner list
        # first index contains the Predicate class
        # second index contains whether to NOT the Predicate
        # the remaining indices are parameters for the Predicate
        pred_list.append([pred_class,
                          random.random() < not_p,
                          *[random.choice(pp) for pp in pred_param_space]]) 

      clauses.append(pred_list)
    
    return clauses
