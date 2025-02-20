import jax
import flax
import jax.numpy as jnp
from brax.training.types import PRNGKey
from typing import Tuple
import functools

@flax.struct.dataclass
class MemoryBankState:
    sa_bank: jnp.ndarray
    g_bank: jnp.ndarray
    insert_position: jnp.ndarray
    num_samples: int
    key: PRNGKey
    
class MemoryBank():
    def __init__(self, memory_bank_size: int, feature_dim: int, batch_size: int):
        self.size = memory_bank_size
        self.feature_dim = feature_dim
        self.batch_size = batch_size # number of samples to sample from the memory bank
    
    def init(self, key):
        return MemoryBankState(
            sa_bank=jnp.zeros((self.size, self.feature_dim)),
            g_bank=jnp.zeros((self.size, self.feature_dim)),
            insert_position=jnp.zeros((), dtype=jnp.int32),
            num_samples=0,
            key=key
        )
    
    def insert(self, state: MemoryBankState, sa_repr: jnp.ndarray, g_repr: jnp.ndarray) -> MemoryBankState:
        batch_size = sa_repr.shape[0]
        indices = jnp.mod(state.insert_position + jnp.arange(batch_size), state.sa_bank.shape[0])
        new_sa_bank = state.sa_bank.at[indices].set(sa_repr)
        new_g_bank = state.g_bank.at[indices].set(g_repr)
        new_insert_position = jnp.mod(state.insert_position + batch_size, state.sa_bank.shape[0])
        new_num_samples = jnp.minimum(state.num_samples + batch_size, self.size)
        return state.replace(sa_bank=new_sa_bank, g_bank=new_g_bank, insert_position=new_insert_position, num_samples=new_num_samples)

    
    def sample(self, state: MemoryBankState) -> Tuple[MemoryBankState, Tuple[jnp.ndarray, jnp.ndarray]]:
        key, sample_key = jax.random.split(state.key)
        num_samples = self.batch_size # can do a multiplier here later
        indices = jax.random.choice(sample_key, state.sa_bank.shape[0], shape=(num_samples,), replace=False)
        # Return updated state and samples
        return state.replace(key=key), (state.sa_bank[indices], state.g_bank[indices])