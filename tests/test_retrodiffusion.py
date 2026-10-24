"""Retro Diffusion client with the HTTP layer mocked: no network, no key needed.  uv run python -m unittest"""
import base64, io, os, tempfile, unittest
from unittest import mock

from PIL import Image

from megaloman import retrodiffusion as rd, store


def _png():
    b = io.BytesIO(); Image.new('RGB', (16, 16), 'red').save(b, 'PNG'); return b.getvalue()


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = [mock.patch.object(store, 'CACHE', self.tmp.name), mock.patch.object(store, 'ROOT', self.tmp.name)]
        for p in self.p: p.start()

    def tearDown(self):
        for p in self.p: p.stop()
        self.tmp.cleanup()

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(rd.MissingKey, 'set RETRO_DIFFUSION_API_KEY'):
                rd.generate(rd.Ledger(), prompt='x', style='rd_pro__default', width=16, height=16)

    def test_task_poll_cache_and_cost(self):
        img = base64.b64encode(_png()).decode()
        calls = []

        def http(method, path, body=None, key=None, timeout=300):
            calls.append((method, path, body, key))
            if method == 'POST':
                return {'status': 'accepted', 'task_id': 't1'}
            return {'status': 'completed', 'result': {'base64_images': [img], 'balance_cost': 0.18}}

        with mock.patch.dict(os.environ, {'RETRO_DIFFUSION_API_KEY': 'rdpk-test'}), \
             mock.patch.object(rd, '_http', http), mock.patch.object(rd.time, 'sleep'):
            led = rd.Ledger(budget=1)
            kw = dict(prompt='snipe', style='rd_pro__default', width=16, height=16, seed=1,
                      reference_images=[_png()], input_palette=_png())
            imgs, cost, cached = rd.generate(led, **kw)
            self.assertEqual((len(imgs), cost, cached), (1, 0.18, False))
            self.assertEqual(calls[0][3], 'rdpk-test')
            self.assertEqual(calls[1][1], '/inferences/tasks/t1')
            body = calls[0][2]
            self.assertEqual(body['prompt_style'], 'rd_pro__default')
            self.assertIn('input_palette', body); self.assertEqual(len(body['reference_images']), 1)
            n = len(calls)
            imgs2, cost2, cached2 = rd.generate(led, **kw)
            self.assertTrue(cached2); self.assertEqual(len(calls), n); self.assertEqual(imgs2, imgs)
            self.assertAlmostEqual(led.spent, 0.18)

    def test_budget(self):
        led = rd.Ledger(budget=0.1)
        with self.assertRaises(rd.BudgetExceeded):
            led.check(rd.price('rd_pro__default'))


if __name__ == '__main__':
    unittest.main()
