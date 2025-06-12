import tempfile
import unittest

import numpy as np

from transformers import FlaxQwen3ForCausalLM, Qwen3Config, Qwen3ForCausalLM, set_seed
from transformers.testing_utils import require_flax, require_torch


if require_flax.is_flax_available():
    import jax.numpy as jnp

if require_torch.is_torch_available():
    import torch


@require_flax
@require_torch
class Qwen3FlaxTorchEquivalenceTest(unittest.TestCase):
    def test_flax_and_torch_equivalence(self):
        seed = 0
        set_seed(seed)

        config = Qwen3Config(
            vocab_size=99,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=37,
            rms_norm_eps=1e-6,
        )

        input_ids = np.arange(4)[None, :]

        pt_model = Qwen3ForCausalLM(config)
        pt_model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_model.save_pretrained(tmpdir)
            fx_model, fx_params = FlaxQwen3ForCausalLM.from_pretrained(tmpdir, from_pt=True, seed=seed)

        with torch.no_grad():
            pt_logits = pt_model(torch.tensor(input_ids)).logits.cpu().numpy()
        fx_logits = fx_model(jnp.array(input_ids), params=fx_params).logits

        self.assertTrue(np.allclose(pt_logits, np.array(fx_logits), atol=1e-4))

        pt_gen = pt_model.generate(torch.tensor(input_ids), do_sample=False, max_new_tokens=2)
        fx_gen = fx_model.generate(jnp.array(input_ids), params=fx_params, do_sample=False, max_new_tokens=2).sequences

        np.testing.assert_array_equal(pt_gen.cpu().numpy(), np.array(fx_gen))


if __name__ == "__main__":
    unittest.main()
