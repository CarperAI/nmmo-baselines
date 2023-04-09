from openelm.mutation_model import DiffModel, MutationModel, PromptModel
from openelm.configs import DiffModelConfig, ELMConfig, PromptModelConfig, EnvConfig, MAPElitesConfig
from openelm import ELM
from openelm.environments import ENVS_DICT
from .environment import NMMOConfig, NMMOTask, NMMO


config = ELMConfig()
config.env = NMMOConfig()
config.qd = MAPElitesConfig()
config.model = PromptModelConfig()
config.batch_size = 8
config.model.model_path = "Salesforce/codegen-2B-mono"

ENVS_DICT["NMMO"] = NMMO

elm = ELM(config)
elm.run(init_steps = 5, total_steps = 20)