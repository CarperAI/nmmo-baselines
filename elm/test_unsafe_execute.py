import functools
from openelm.sandbox.server.sandbox_codex_execute import ExecResult, unsafe_execute

from nmmo.entity.entity import Entity
from nmmo.systems.item import Item
from nmmo.systems import item
from .environment import MockRealm

prompts = [
   """
def gold_pickaxe_task(entity: Entity):
    # Task to hoard 30 gold, have health above 50 and own a level 3 pickaxe
    hoard_30_gold = HoardGold(gold=30)
    own_level_3_pickaxe = OwnItem(item.Pickaxe, min_level=3)

    return AND(hoard_30_gold, own_level_3_pickaxe)
    #return True
    """     
]


e = functools.partial(
    unsafe_execute,
    timeout=5.0,
    debug=True,
)
realm = MockRealm()
it = item.Pickaxe(realm, 10)
entity_id = 123
population_id = 11
entity = Entity(realm, (10,20), entity_id, "name", "color", population_id)
print(e(prompts[0], timeout=5.0, args={"entity":entity}, debug=True))