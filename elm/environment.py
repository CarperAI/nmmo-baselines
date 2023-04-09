from openelm.mutation_model import DiffModel, MutationModel, PromptModel
from openelm.configs import DiffModelConfig, ELMConfig, PromptModelConfig, EnvConfig, MAPElitesConfig
from openelm import ELM
from openelm.environments import BaseEnvironment, Genotype
from openelm.utils.code_eval import pool_exec_processes, type_check
from nmmo.task.predicate.core import *
import nmmo
from nmmo.datastore.numpy_datastore import NumpyDatastore
from nmmo.entity.entity import Entity, EntityState
from nmmo.systems.item import ItemState
from typing import Generic, Optional, Type, TypeVar, Union
import numpy as np
uniq_predicates = ["TickGE","StayAlive","AllDead","EatFood","DrinkWater","CanSeeTile","CanSeeAgent","OccupyTile","DistanceTraveled","AllMembersWithinRange","ScoreHit","ScoreKill","AttainSkill","InventorySpaceGE","OwnItem","EquipItem","FullyArmed","ConsumeItem","GiveItem","DestroyItem","HarvestItem","HoardGold","GiveGold","ListItem","EarnGold","BuyItem","SpendGold","MakeProfit"]
from dataclasses import dataclass, field



@dataclass
class NMMOConfig(EnvConfig):
    env_name:str = "NMMO"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            # Unique_predicates, length of the task, difficulty
            [0, 20],
            [0, 1000],
            [0, 10],
        ]
    )
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 1
    crossover: bool = False

class MockRealm:
  def __init__(self):
    self.config = nmmo.config.Default()
    self.config.PLAYERS = range(100)
    self.datastore = NumpyDatastore()
    self.items={}
    self.datastore.register_object_type("Entity", EntityState.State.num_attributes)
    self.datastore.register_object_type("Item", ItemState.State.num_attributes)

Phenotype = Optional[np.ndarray]
class NMMOTask(Genotype):
    def __init__(self, program_str: str):

        # to check if the task is valid
        if self.check_valid(program_str):
            self.valid = True
            self.program_str: str = program_str
            self.morphology = {}
            self.morphology["predicates"] = self._count_predicates(program_str) 
            self.morphology["length"] = len(program_str)
            self.morphology["lines"] = program_str.count(r"\n")
        else:
            self.valid = False

    def evaluate(self) -> float:
        # how to evaluate the fitness of a task? (time taken for the baseline RL algo to solve?)

        self._fitness = 1/len(self.program_str)

        return self._fitness
    
    def check_valid(self, program_str: str):
        # additional checks if tasks are correct
        return True

    def __str__(self) -> str:
        return self.program_str
    
    def _count_predicates(self, task_str):
        predicates = 0
        for i in uniq_predicates:
            if i in task_str:
                predicates+=1
        return predicates
        
    def to_phenotype(self) -> Optional[Phenotype]:
        # phenotypes of the task?
        # creating a dummy version, string length of the task, unique predicates, difficulty of the task,
        if self.valid:
            return np.array(
                [
                    self.morphology["predicates"],
                    self.morphology["length"],
                    self.morphology["lines"]
                ]
            )
        else: 
            return None



    @property
    def fitness(self) -> Optional[float]:
        return self._fitness

class NMMO(BaseEnvironment[NMMOTask]):
    def __init__(
        self,
        config: NMMOConfig,
        mutation_model: PromptModel,
    ) -> None:

        self.config: NMMOConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model: PromptModel = mutation_model
        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]


    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        
        prompt_str = ""
        import_str = ""
        if code_batch is None:
            # first random task, just giving the dummy one for reference
            prompt_str = """
            """
            import_str = """# Mutate tasks to create different Values
def gold_pickaxe_task(entity: Entity):
    # Task to hoard 30 gold, have health above 50 and own a level 3 pickaxe
    hoard_30_gold = HoardGold(gold=30)
    own_level_3_pickaxe = OwnItem(item.Pickaxe, min_level=3)
    return AND(hoard_30_gold, own_level_3_pickaxe)"""
        else:
            if isinstance(code_batch, list):
                prompt_str += code_batch[0]
            elif isinstance(code_batch, str):
                prompt_str += code_batch
        return {"prompt": prompt_str, "template": import_str}

    def generate_programs(self, code_batch: list[dict[str, str]]) -> list[NMMOTask]:
        
        local_scope_exec: bool = self.config.instruction != 0

        generated_tasks = self.mutation_model.generate_programs(
            code_batch, local_scope_exec
        )
        realm = MockRealm()
        entity_id = 123
        population_id = 11
        entity = Entity(realm, (10,20), entity_id, "name", "color", population_id)


        results = pool_exec_processes(
            generated_tasks,
            timeout=5.0,
            args={"entity":entity},
            debug=False
        )
        result_list: list = []
        for i, result in enumerate(results):
            try:
                if isinstance(result, AND) or isinstance(result, OR) or isinstance(result, Predicate): 
                    print(generated_tasks[i])
                    result_list.append(generated_tasks[i])
            except Exception as e:
                print(type(e))


        return [NMMOTask(t) for t in result_list]

    def random(self) -> list[NMMOTask]:
        program_list = [self.construct_prompt() for _ in range(8)]
        new_tasks = self.generate_programs(program_list)
        return new_tasks

    def mutate(self, task_list: list[NMMOTask]) -> list[NMMOTask]:
        print("mutating")
        tasks = [sr.program_str for sr in task_list]
        program_list = list(map(self.construct_prompt, tasks))
        new_tasks = self.generate_programs(program_list)

        return new_tasks

    def fitness(self, x: NMMOTask) -> float:
        print("checking fitness")
        if x.valid:
            return x.evaluate()
        else:
            return -np.inf
