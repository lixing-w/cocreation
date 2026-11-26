"""Generation grouping must preserve the DAG and reduce resident model swaps."""
import itertools
import random
import unittest
from scripts.dialogue_generation.materialize import materialization_order, execution_key
from cocreation import MiniMaxH3, H3Runtime
from cocreation.h3 import build_graph, TURBO_LORA


def task(backend=None, *deps):
    return dict(operation='reference_generation' if backend else 'copy',
                backend=backend, dependencies=list(deps))


def switches(order, tasks):
    keys = [execution_key(tasks[p]) for p in order if execution_key(tasks[p])]
    return sum(a != b for a, b in zip(keys, keys[1:]))


class MaterializationOrderTests(unittest.TestCase):
    def check_order(self, tasks):
        result = materialization_order(tasks)
        self.assertEqual(set(result), set(tasks))
        positions = {p: i for i, p in enumerate(result)}
        for p, row in tasks.items():
            for dep in row['dependencies']:
                self.assertLess(positions[dep], positions[p])
        return result

    def test_independent_images_run_before_their_videos(self):
        tasks = dict(i1=task('flux'), v1=task('minimax_h3', 'i1'),
                     i2=task('flux'), v2=task('minimax_h3', 'i2'))
        self.assertEqual(self.check_order(tasks), ['i1', 'i2', 'v1', 'v2'])

    def test_cpu_dependencies_do_not_interrupt_image_phase(self):
        tasks = dict(i1=task('flux'), v1=task('minimax_h3', 'i1'),
                     copy=task(None, 'i1'), i2=task('flux', 'copy'))
        order = self.check_order(tasks)
        self.assertLess(order.index('i2'), order.index('v1'))
        self.assertEqual(switches(order, tasks), 1)

    def test_starting_model_is_chosen_to_avoid_an_extra_phase(self):
        tasks = dict(v1=task('minimax_h3'), i1=task('flux'), v2=task('minimax_h3', 'i1'))
        self.assertEqual(switches(self.check_order(tasks), tasks), 1)

    def test_video_to_image_dependency_forces_the_necessary_switch(self):
        tasks = dict(i1=task('flux'), v1=task('minimax_h3', 'i1'),
                     i2=task('flux', 'v1'), v2=task('minimax_h3', 'i2'))
        self.assertEqual(self.check_order(tasks), list(tasks))

    def test_two_model_schedule_matches_small_dag_optimum(self):
        rng = random.Random(16)
        for _ in range(24):
            tasks = {i: task(rng.choice(['flux', 'minimax_h3', None]),
                             *(j for j in range(i) if rng.random() < .25)) for i in range(6)}
            valid = [p for p in itertools.permutations(tasks)
                     if all(p.index(dep) < p.index(i) for i in tasks for dep in tasks[i]['dependencies'])]
            self.assertEqual(switches(self.check_order(tasks), tasks),
                             min(switches(p, tasks) for p in valid))

    def test_explicit_turbo_uses_ref2va_and_dual_schedule_sampler(self):
        request = MiniMaxH3(runtime=H3Runtime(turbo=True))('A peaceful scene.', seconds=4, dry_run=True)
        self.assertEqual(request['options']['seconds'], 4)
        self.assertEqual(request['options']['frames'], 107)
        with self.assertRaises(ValueError):
            MiniMaxH3(runtime=H3Runtime(turbo=True))('Scene.', seconds=3.99, dry_run=True)
        self.assertEqual(request['options']['steps'], 6)
        self.assertEqual(request['options']['variant'], 'ref2va')
        graph = build_graph(request, [])
        self.assertEqual(graph['turbo_lora']['inputs']['lora_name'], TURBO_LORA)
        self.assertEqual(graph['turbo_lora']['inputs']['strength'], 1.)
        self.assertFalse(graph['turbo_lora']['inputs']['low_vram'])
        self.assertEqual(graph['sample']['class_type'], 'SamplerCustomAdvanced')
        self.assertEqual(graph['sigmas']['inputs']['steps'], 6)
        self.assertEqual(graph['sigmas']['inputs']['scheduler'], 'simple')
        self.assertEqual(graph['sample']['inputs']['sampler'], ['turbo_sampler', 0])
        plain = MiniMaxH3(runtime=H3Runtime(turbo=False))('Scene.', dry_run=True)
        self.assertEqual(build_graph(plain, [])['sample']['class_type'], 'KSampler')


if __name__ == '__main__':
    unittest.main()
